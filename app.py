from database import DatabaseFile
from query import Query, GoogleScholarQuery
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import FLASK_HOST, DEEPL_AUTH_KEY
from chatbot import Chatbot, MockChatbot, ChatbotKnowledgeBuilder
from deepl import Translator
from utils import debug
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
name: str | None = None
database: DatabaseFile | None = None
chatbot_knowledge_builder: ChatbotKnowledgeBuilder | None = None 
chatbot: Chatbot | None = None
translator: Translator = Translator(DEEPL_AUTH_KEY)
model_name: str | None = None

model = None
tokenizer = None

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate the text from the specified source language to the specified target language.

    Args:
        text (str): The text to be translated.
        source_lang (str): The source language.
        target_lang (str): The target language.
    """
    debug(f'Translating text with {len(text)} characters from `{source_lang}` to `{target_lang}`.', 'DEEPL')

    # Translate the `text` from `source_lang` to `target_lang`.
    return translator.translate_text(text=text, target_lang=target_lang).text

def initialize_chatbot_knowledge(rname: str) -> None:
    """
    Initializes the chatbot knowledge in relation to the researcher's name.

    Args:
        rname (str): The researcher's name used to initialize the chatbot knowledge.

    Note:
        ... This function could be used again to reset the chatbot knowledge to another
        researcher. Therefore, if the following command is executed:
        >>> initialize_chatbot_knowledge('Researcher A')
        Then the chatbot initialize its knowledge with information from `Researcher A`.
        Nevertheless, if afterwards the following command is executed:
        >>> initialize_chatbot_knowledge('Researcher B')
        Then the chatbot initialize its knowledge with information from `Researcher B`,
        discarding all previous knowledge obtained from `Researcher A`.

        ... There are some internal decisions tackled of why this is done. Mainly, this is done
        to prevent the LLM model to cross (or relate) information from both researchers to answer
        about the information asked of one of them.
    """
    # Bound to the global variables.
    global name
    global database
    global chatbot_knowledge_builder
    global chatbot
    global model
    global tokenizer

    # Fetch the researcher information by its name.
    info = database.fetch_by_name(rname)
    
    # Check if the researcher has not been found in the database.
    if info is None:
        # Query the researcher information online.
        info = query.query_by_name(rname)
        
        # The researcher has not been found online.
        if info is None:
            return 'Não foi possível encontrar este pesquisador em nossa base de dados local nem em bases de dados online como o Google Scholar.'

        # The researcher information has been found online. Let's cache it.
        database.cache_info(info)
    
    # Initialize the researcher's name (globally).
    name = rname

    debug(f'Variable `name` set to {rname}.', 'GLOBAL')

    # Chatbot initialization.
    chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
    chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)
    # chatbot = MockChatbot()

    return 'O que deseja perguntar?'

@app.route('/')
def index():
    """
    GET `/` Endpoint.

    This endpoint is responsible to receive the incoming client-side connections
    and rendering the HTML template site to the clients.
    """
    content = {
        'version': 'v1.0.0',
    }

    return render_template('index.html', **content)

@app.route('/response', methods=['POST'])
def response():
    """
    POST `/response` Endpoint.

    This endpoint is responsible for receive the `message` from the client
    and answer it using a AI-generated message using a LLM (Large Language Model).
    """
    # Bound to the global variables.
    global chatbot_knowledge_builder
    global chatbot
    global name

    # Retrieve the client's message.
    message = request.json.get("message")
    
    if name is None:
        # Initialize the chatbot knowledge, if no researcher has been previously specified.
        return jsonify(response=initialize_chatbot_knowledge(rname=message))
    
    # Message translated to English language.
    message_en = translate_text(message, source_lang='PT-BR', target_lang='EN-US')

    # Generate the AI message.
    response = chatbot.query(message_en)

    # Response translated to Brazilian Portuguese language.
    response_ptbr = translate_text(response, source_lang='EN-US', target_lang='PT-BR')

    # Add the AI-generated message as a recent knowledge.
    chatbot_knowledge_builder.add_response(response_ptbr)

    return jsonify(response=response_ptbr)

@app.route('/select_model', methods=['POST'])
def select_model():
    # Bound to the global variables.
    global model
    global tokenizer
    global model_name

    # Retrieve the selected model's name.
    mname = request.json.get("model_name")

    debug(f'Loading the selected model `{mname}`...', 'MODEL SELECTOR')

    try:
        # LLM Model and Tokenizer.
        model = AutoModelForCausalLM.from_pretrained(
            mname,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(mname)
    except Exception as e:
        print(e)
        debug(f'Model `{mname}` could not be loaded.', 'MODEL SELECTOR')
        return jsonify(status=-1, message='Não foi possível encontrar este modelo.')

    # Update the model name globally.
    model_name = mname

    debug(f'Variable `model_name` set to {model_name}.', 'GLOBAL')

    return jsonify(status=0, message='Modelo encontrado com sucesso!')

@app.route('/reset_chatbot_knowledge', methods=['POST'])
def reset_chatbot_knowledge():
    # Bound to the global variables.
    global name
    global chatbot_knowledge_builder
    global chatbot

    # Reset variables.
    name = None
    chatbot_knowledge_builder = None
    chatbot = None

    debug('Chatbot knowledge has been reset.', 'RESET CHATBOT KNOWLEDGE')

    return jsonify(status=0, message='Chatbot resetado com sucesso!')

if __name__ == "__main__":
    # Database initialization.
    database = DatabaseFile(root='dataset.json')
    database.load_database_info()

    # Query initialization.
    query: Query = GoogleScholarQuery()

    # Run the app.
    app.run(debug=True, host=FLASK_HOST)