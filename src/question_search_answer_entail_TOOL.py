from haystack.document_stores import InMemoryDocumentStore
from newspaper3k_haystack import newspaper3k_scraper
from duckduckgo_search import DDGS
from haystack.pipelines import Pipeline
from haystack.nodes import PreProcessor
from haystack.nodes import EmbeddingRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack_entailment_checker import EntailmentChecker
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class question_search_answer_entail():
    def __init__(self,TOP_LINKS = 25,document_store = None,retriever=None):

        #we have the ability of adding our new knowledge to a bigger db
        if document_store is None:
            self.document_store = InMemoryDocumentStore()
        else:
            self.document_store = document_store

        ##Indexing pipeline stuff:
        self.scraper = newspaper3k_scraper()
        self.processor = PreProcessor(
            clean_empty_lines=False,
            clean_whitespace=False,
            clean_header_footer=False,
            split_by="sentence",
            split_length=30,
            split_respect_sentence_boundary=False,
            split_overlap=0
            )

        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_node(component=self.scraper, name="scraper", inputs=['File'])
        self.indexing_pipeline.add_node(component=self.processor, name="processor", inputs=['scraper'])
        self.indexing_pipeline.add_node(component=self.document_store, name="document_store", inputs=['scraper'])

        #Extraction pipeline stuff:
        if retriever == None:
            self.retriever = EmbeddingRetriever(
                document_store=self.document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",use_gpu=True,devices=[torch.device("mps")]
            )
        else:
            self.retriever = retriever
        self.reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True,devices=[torch.device("mps")])
        self.extractive_pipe = ExtractiveQAPipeline(self.reader, self.retriever)

        #To entail sentence generation
        self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

        #Entailment checker stuff
        self.entailment_checker = EntailmentChecker(
        model_name_or_path = "microsoft/deberta-v2-xlarge-mnli",
        entailment_contradiction_threshold = 0.5,use_gpu=True)

        self.entailment_check_pipeline = Pipeline()
        self.entailment_check_pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        self.entailment_check_pipeline.add_node(component=self.entailment_checker, name="EntailmentChecker", inputs=["Retriever"]
        )

    
    def run(self,question:str)->str:
        #get links to scrape
        with DDGS() as ddgs:
            results = list(ddgs.text(question, safesearch='Off'))
        
        links = [r["href"] for r in results][:25]
        
        #use indexing pipeline to get pages
        self.indexing_pipeline.run_batch(queries = links,
            params={
                "scraper":{
                    "metadata":True,
                    "summary":True,
                }
            })
        
        #create embeddings for each documents so we can later on retrieve them semantically
        self.document_store.update_embeddings(self.retriever)
        initial_answers = self.extractive_pipe.run(query=question,params={"Retriever": {"top_k": 20}, "Reader": {"top_k": 5}})

        initial_answers = initial_answers["answers"]

        #check using the entailment checker node each answer:
        answersANDentails = []
        for anw in initial_answers:
            #generate sentence to entail
            inputs = self.t5_tokenizer(f"Give a sentence with the statement of the answer to the question.\n Question: {question}\n Answer:{anw.answer} \nSentence:", return_tensors="pt")
            outputs = self.t5_model.generate(**inputs)
            sentence = self.t5_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            answersANDentails.append(self.entailment_check_pipeline.run(query=sentence))
        
        #We'll return in string form the question, and each of the generated answers together with their entailment info.
        question_output = "Question: "+question + "\nAnswers: "
        for answer in answersANDentails:
            question_output += " - "+answer["query"] + " " + str(answer["aggregate_entailment_info"])
        
        return question_output