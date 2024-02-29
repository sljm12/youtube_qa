""""""
import gradio as gr, os, glob, subprocess,time, datetime,urllib.parse,requests
from langchain.schema.document import Document
from langchain_vttsplitter.vttsplitter import VTTSplitter
from langchain_vttsplitter.loader import YoutubeSubtitleLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from openai import OpenAI

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class VectorDB:
    """VectorDB management class"""
    def __init__(self):
        self.db = None

    def query(self, question):
        """Query VectorDB for the question"""
        ans=self.db.similarity_search_with_score(question)
        context="\n".join([i[0].page_content+'\n' for i in ans])
        #return pipeline(QA_PROMPT.format(question=question, context=context)), ans
        return question

    def load_db(self, url):
        """Reloads DB with the Vectors from the URL"""
        if self.db is not None:
            self.db.delete_collection()

        msg = ""
        try:
            vtt_doc = YoutubeSubtitleLoader(youtube_url=url,language='en').load()
            docs = VTTSplitter().split_text_docs(vtt_doc[0])
            self.db = Chroma.from_documents(docs, embeddings,collection_metadata={"url":url})
            done_fmt = "Done: {url} {date} Collection Count: {collection_count}"
            msg = done_fmt.format(url=url,
                                date=datetime.datetime.now().isoformat()
                                ,collection_count=str(self.db._collection.count()))

        except ValueError:
            msg = "Error loading "+url

        return msg

class LLM:
    def __init__(self, url):
        self.client = OpenAI(base_url=url, api_key="not-needed")
        self.prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
"""

    def execute_query(self, question, context):
        prompt = self.prompt_template.format(question=question, context=context)
        completion = self.client.chat.completions.create(
        model="local-model", # this field is currently unused
        messages=[    
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        )

        return completion.choices[0].message.content

llm = LLM("http://localhost:1234/v1")

iframe_yt = """
<tr>
<td>
<iframe id="ytplayer" type="text/html" width="640" height="360"
  src="{yt_url}&cc_load_policy=1"
  frameborder="0" ></iframe>
</td>
<td>
{transcript}
</td>
</tr>
"""

table_yt = """
<table>
{rows}
</table>
"""

vdb = VectorDB()

def download(u):   
    return vdb.load_db(u)

def extract_embed_url(link,start_time,end_time):
    qs=urllib.parse.urlparse(link).query
    params=urllib.parse.parse_qs(qs)
    return "https://youtube.com/embed/{id}?start={start}&end={end}".format(id=params["v"][0],
                                                                            start=int(start_time),
                                                                            end=int(end_time))

def return_html_iframe(links, transcripts):
    rows = "\n".join([iframe_yt.format(yt_url=l,transcript=transcripts[i]) for i,l in enumerate(links)])
    return table_yt.format(rows=rows)

def query_click(query):
    
    ans=vdb.db.similarity_search_with_score(query)
    context="\n".join([i[0].page_content+'\n' for i in ans])
    transcripts = [i[0].page_content for i in ans]
    video_ids = [extract_embed_url(i[0].metadata["url"],i[0].metadata["start_time"],i[0].metadata["end_time"]) for i in ans]
    return llm.execute_query(query,context),return_html_iframe(video_ids, transcripts)
    #return query,return_html_iframe(video_ids, transcripts)

if __name__ == "__main__":

    with gr.Blocks() as demo:
        with gr.Row():
            youtube=gr.Textbox(label="Youtube URL", value ="https://www.youtube.com/watch?v=LCcWWbx6pXU", scale=3)
            load_youtube=gr.Button(value="Load Youtube", scale=1)


        youtube_output=gr.Textbox(label="Loading")
        load_youtube.click(download,inputs=[youtube],outputs=[youtube_output])

        with gr.Row():
            query_text=gr.Textbox(label="Query",value="What are the initiatives in the rally?",scale=3)
            query_btn=gr.Button(value="Submit Query", scale=1)


        output=gr.Textbox(lines=7,label="Output")

        html_1=gr.HTML()
        query_btn.click(query_click,inputs=[query_text],outputs=[output,html_1])

    demo.launch(debug=True)
