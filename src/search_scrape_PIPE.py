from haystack.nodes import EntityExtractor
from haystack.pipelines import Pipeline
from haystack.nodes import PreProcessor, BM25Retriever, FARMReader
import torch
from newspaper3k_haystack import newspaper3k_scraper
from duckduckgo_search import DDGS

class search_scrape_pipeline():
    def __init__(self,document_store):
        
        self.document_store = document_store

        self.scraper = newspaper3k_scraper()
        
        self.entity_extractor = EntityExtractor(model_name_or_path="dslim/bert-base-NER",devices=[torch.device("mps")],flatten_entities_in_meta_data=True)

        self.processor = PreProcessor(
            clean_empty_lines=False,
            clean_whitespace=False,
            clean_header_footer=False,
            split_by="sentence",
            split_length=30,
            split_respect_sentence_boundary=False,
            split_overlap=0 #try changing this in the future :)
        )

        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_node(component=self.scraper, name="scraper", inputs=['File'])
        self.indexing_pipeline.add_node(component=self.processor, name="processor", inputs=['scraper'])
        self.indexing_pipeline.add_node(self.entity_extractor, "EntityExtractor", ["processor"])
        self.indexing_pipeline.add_node(component=self.document_store, name="document_store", inputs=['EntityExtractor'])

    def run(self,query:str,topk=25)->list:
        #get links to scrape
        with DDGS() as ddgs:
            results = list(ddgs.text(query, safesearch='Off'))
        
        links = [r["href"] for r in results][:topk]

        #before loading any of the links we want to check if that page has already been scraped in our db
        
        #get all scraped urls
    
        scraped_urls = [doc.meta["url"] for doc in self.document_store.get_all_documents_generator()]
        scraped_urls = set(scraped_urls) #for faster searching

        for j,lk in enumerate(links):
            if lk in scraped_urls:
                print(f"Already in DB: {lk}")
                links.pop(j)

        #use indexing pipeline to get pages
        self.indexing_pipeline.run_batch(queries = links,
            params={
                "scraper":{
                    "metadata":True,
                    "summary":True,
                }
            })
        
        #return new links
        return links