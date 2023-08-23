# RAG_TouristMap_generator
An experiment using haystack to extract and synthesize a small summary of touristically interesting places by combining  different NLP techniques. Check it out here  ☞ ͡°◡ ͡°☞ [here!](https://haradai.github.io/RAG_TouristMap_generator/)

# Process:
This was the process used to generate the map. I wanted to share it here for my own record and in case anyone finds it useful and might want to share in a similar project what worked better or worse. If you see things that could be done better pleasee say something! :)

## 1. Scrape article webpages 
Crawled many pages using the newspaper3k-haystack crawler node and filtered urls by "norway". These documents where split, with some overlap and then created an embedding and saved onto the document store.

## 2. Extracting places to research on
 Using NER node from haystack, NER metadata was created for every document. Then the entities with tag LOC were gathered to start asking our system one by one for each of them.

## 3. Filtering places
This part was a bit tricky because for start you can check if a place exists by using the geopy library for example. But then you get words like Library, School... These words will return some random coordinate but without any context do not refer to any specific place, this is why first a prompt node was used so gpt could filter for us.

The question was: 

>  Does the word {query} refer to a general concept? Answer only Yes or No.

 Then answer parsed and we have some initial filter.
 After this just look if the coordinates can be found using a .csv file downloaded from the internet and if not use geopy search, filtering for  "NO" (norway).

## 4. FAQ generation
This part is literally just a RAG QA pipeline where chatgpt-3.5-turbo acts as the end answerer.
To make answers better I played around for a while with different retrieving pipelines and this is the best I could do:

- Embedding retriever "sentence-transformers/multi-qa-mpnet-base-dot-v1" model. 
- topPsampler 
- DiversityRanker

Using this pipeline then a set of questions are asked and so on.

## 5. Summary generation
Using the FAQ generation retrieving pipeline didn't work very well because when creating a summary I wanted to retrieve a bunch of documents on very open questions.  The results of documents I was getting didn't seem very useful.

So as an experiment I tried the following:

1. For each document generate questions using the question generator node
2. Create a parallel document store and save as content the generated questions while keeping the original text for each document in the metadata.

With this now, and using a simple sentence to sentence similarity embedding, not specifically for qa I queried documents like this: 

> f"What is {place}famous for?\nWhat are the best things to do in
> {place}?\nWhat activities can I do in {place}?What are must see places
> in {place}?"

This, seemed to return better documents, not really sure if was better and I don't know how to more or less measure it.

Finally, asked again with the retrieved documents gpt the following:
> Elaborate a 100 words description about {query}, use your own words
> but it should be truthfully based solely on the given documents.
> 
> Try to add as many different information as possible but avoid giving
> too many concrete details. Be diverse, the description should be
> broad. Remember to keep it short, maximum 100 words.
> 
> Use a touristic guide tone. For everything you say, properlly cite
> only the documents where the information was extracted from using
> Document[number] notation. If multiple documents contain the answer,
> cite those documents like as stated in Document[number],
> Document[number], ... e.g. [1],[2]
> 
> {join(documents, delimiter=new_line, pattern=new_line+'Document[$idx]:
> $content', str_replace={new_line: ' ', '[': '(', ']': ')'})}
> 
> Answer:

Seemed to do something, summaries where nice thought the references sometimes point to random pages not related to what was stated. (Not sure how to improve this) gpt4?

## 6. Tagging places

Finally I wanted to show on the map a different color for different types of places:
{"culinary":"orange",
"nature":"green",
"cultural":"blue",
"city":"purple",
"other":"gray"}

For this I just used a zero-shot classifier from hugging face on the summary generated. For now "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", but wanted to try other models because it didn't always work great. I think it doesn't do well on long texts.

## 7. Generating the map
For the map generation the library in python *"folium*" was used to generate the map html file. Then this is embedded onto another simple html page partially created using chat gpt :)