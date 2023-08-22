from haystack.pipelines import DocumentSearchPipeline

class simple_search():
    def __init__(self,document_store,retriever):
        self.document_store = document_store
        self.pipeline = DocumentSearchPipeline(retriever=retriever)

    def run(self, query:str, filter:str, topk=5)->str:
        result = self.pipeline.run(
            query=query,
            params={
                "Retriever": {
                    "top_k": topk,
                    "filters": {
                        'entity_words': [filter]
                    }
                }
            }
        )

        toanswer = f"Query: {query}\n"

        for i, doc in enumerate(result["documents"]):
            toanswer += f"({i+1})\n{doc.content}\n"
        
        return toanswer