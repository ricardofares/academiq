import torch

from database import DatabaseFile
from query import Query, GoogleScholarQuery
from transformers import AutoModelForCausalLM, AutoTokenizer
from constants import DATASET_CONFIG, PROMPT_BASE_INFORMATION, MODEL_NAME
from abc import ABC, abstractmethod

class ChatbotKnowledgeBuilder:

    def __init__(self, info: dict):
        self._prior_knowledge = self._mount_prior_knowledge(info)  # Chatbot's prior knowledge (base context).
        self._responses = []  # List storing the responses of the chatbot.

    def add_response(self, response: str) -> None:
        # Append the responses.
        self._responses.append(response)

    def build_query(self, text: str) -> list[dict]:
        """
        Mounts the query of the LLM model. This query buidls the so called posterior knowledge, which is composed
        by the prior knowledge (base context of the LLM model) with the query `text` entered by the user.

        Args:
            text (str): The query of the user.

        Returns:
            list[dict]: The posterior knowledge of the LLM model.
        """
        # Chatbot's posterior knowledge (after user entered a text).
        return [
            *self._prior_knowledge,
            *self._mount_responses_knowledge(),
            { "role": "user", "content": text },
        ]

    def _mount_prior_knowledge(self, info: dict) -> list[dict]:
        """
        Mounts the prior knowledge (base context) of the chatbot.

        Args:
            info (dict): The researcher information.

        Returns:
            list[dict]: The prior knowledge of the LLM model.
        """
        # Base knowledge of the chatbot composed by two prior knowledges. First,
        # the knowledge of the assistant and second the knowledge of the assistant.
        return [
            *self._mount_assistant_base_knowledge(),
            *self._mount_researcher_base_knowledge(info),
        ]

    def _mount_assistant_base_knowledge(self) -> list[dict]:
        """
        Mounts the base knowledge of the LLM model informating it, that it will be an assistant
        for querying information of researchers and professors.

        Notes:
            ... This is the base knowledge of the assistant, it will be sent together with every
                query of the user, therefore, allowing the assistant to have a context and enabling
                it to give satisfactory responses.

        Returns:
            list[dict]: The base knowlege of the LLM model for informating it, that it will
                be an assistant for querying information of researchers and professors.
        """
        return [{ "role": "system", "content": base_knowledge } for base_knowledge in PROMPT_BASE_INFORMATION]
    
    def _mount_researcher_base_knowledge(self, info: dict) -> list[dict]:
        """
        Mounts the base knowledge of the LLM model about the researcher, where `info` contains the
        information of the researcher queried from the database or online (API).

        Notes:
            ... This is the base knowledge of the assistant with relation to the researcher
                being queried.

        Returns:
            list[dict]: The base knowledge of the LLM model containing context information about
                the researcher being queried.
        """
        base_knowledge_list = []

        # Append information about the researcher affiliation.
        base_knowledge_list.append(
            f"{info['name']} is a researcher affiliated to {info['affiliation']}"
        )

        # Append information about the number of citations.
        base_knowledge_list.append(
            f"{info['name']} is a researcher that has been cited {info['citedby']} times."
        )

        # Append information about the h-index.
        base_knowledge_list.append(
            f"{info['name']} is a researcher that has h-index {info['h-index']}."
        )

        # Append information about the researcher's keywords or interests.
        base_knowledge_list.append(
            f"{info['name']} keywords are {', '.join(info['keywords'])}."
        )
        base_knowledge_list.append(
            f"{info['name']} interests are {', '.join(info['keywords'])}."
        )

        # Append information about the researcher's coauthors.
        for coauthor in info['coauthors']:
            base_knowledge_list.append(
                f"{info['name']} coauthored with {coauthor['name']} who is affiliated to {coauthor['affiliation']}"
            )

        # Append information about the top-10 most cited publications.
        for publication in info['top10_cited_papers']:
            base_knowledge_list.append(
                f"{info['name']} published the worked entitled {publication['title']} at {publication['citation']} in {publication['pub_year']} having {publication['num_citations']} citations."
            )

        return [{ "role": "system", "content": base_knowledge } for base_knowledge in base_knowledge_list]

    def _mount_responses_knowledge(self) -> list[dict]:
        return [{
            "role": "system",
            "content": f"The assistant answered: {response}"
        } for response in self._responses]

class BaseChatbot(ABC):
    
    @abstractmethod
    def query(self, text: str) -> str:
        """
        Ask the chatbot using the specified text.

        Args:
            text (str): The specified text to be asked to the chatbot.

        Returns:
            (str): The chatbot (AI-generated) response.
        """
        return

class MockChatbot(BaseChatbot):

    def query(self, text: str) -> str:
        """
        Ask the mock chatbot using the specified text.

        Args:
            text (str): The specified text to be asked to the mock chatbot.

        Returns:
            (str): It always returns `Mock Response`.
        
        Notes:
            ... This is the mock chatbot using during the testing/development stage
            in relation to the client-server interaction without focusing in the
            AI-generated resonses.
        """
        return 'Mock Response'

class Chatbot:
    
    def __init__(self, model, tokenizer, knowledge_builder: ChatbotKnowledgeBuilder) -> None:
        self._model = model  # Chatbot's LLM model.
        self._tokenizer = tokenizer  # Chatbot's tokenizer.
        self._knowledge_builder = knowledge_builder  # Chatbot's knowledge builder.

    def query(self, text: str) -> str:
        # Assuming 'messages', 'tokenizer', and 'model' are defined
        text = self._tokenizer.apply_chat_template(
            self._knowledge_builder.build_query(text),
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # Generate the output tokens from the model
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Response.
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    def query_with_perplexity(self, text: str) -> str:
        # Assuming 'messages', 'tokenizer', and 'model' are defined
        text = self._tokenizer.apply_chat_template(
            self._knowledge_builder.build_query(text),
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        # Generate the output tokens from the model
        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # To compute perplexity, get the logits for the generated sequence
        with torch.no_grad():
            outputs = self._model(**model_inputs, labels=model_inputs["input_ids"])
            logits = outputs.logits

        # Flatten the logits and shift tokens to align for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()  # Remove the last token logits (as we don't predict anything after it)
        shift_labels = model_inputs["input_ids"][..., 1:].contiguous()  # Shift input_ids to match next-token prediction

        # Compute the log-likelihood for each token
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)  # Convert logits to log-probabilities
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # Gather the log-prob of the true token

        # Calculate the average log-likelihood
        avg_log_likelihood = token_log_probs.mean()

        # Calculate perplexity
        perplexity = torch.exp(-avg_log_likelihood)
        return perplexity

def main():
    # Database initialization.
    database: DatabaseFile = DatabaseFile(root=DATASET_CONFIG)
    database.load_database_info()

    # Query initialization.
    query: Query = GoogleScholarQuery()

    # LLM Model and Tokenizer.
    model_name = MODEL_NAME
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Requires from the standard input the researcher name.    
    name = input("Insert the name of the researcher: ")
    
    # Fetch the researcher information by its name.
    info = database.fetch_by_name(name)
    
    # Check if the researcher has not been found in the database.
    if info is None:
        # Query the researcher information online.
        info = query.query_by_name(name)
        
        # The researcher has not been found online.
        if info is None:
            print("Researcher could not been find.")
            return

        # The researcher information has been found online. Let's cache it.
        database.cache_info(info)
    
    # Chatbot initialization.
    chatbot_knowledge_builder = ChatbotKnowledgeBuilder(info)
    chatbot = Chatbot(model, tokenizer, chatbot_knowledge_builder)

    while True:
        text = input("Ask something: ")
        
        if text.lower() == 'exit':
            break

        response = chatbot.query(text)

        chatbot_knowledge_builder.add_response(response)

        print(response)

if __name__ == '__main__':
    main()