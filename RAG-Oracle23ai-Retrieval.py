import oracledb
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
print("Successfully imported libraries and modules")

#Declare username and password and dsn (data connection string)
username = "ADMIN"
password = "ArnavTalwani1"
dsn ='''(description= (retry_count=20)(retry_delay=3)(address=(protocol=tcps)(port=1522)(host=adb.us-chicago-1.oraclecloud.com))(connect_data=(service_name=gde2a92f932143d_test23ai_high.adb.oraclecloud.com))(security=(ssl_server_dn_match=yes)))'''

#Connect to the database
try:
    conn23c = oracledb.connect(user=username, password=password, dsn=dsn)
    print("Connection successful!")
except Exception as e:
    print("Connection failed!")

#Retrieval Step 1 - Build the llm, embed_model, and prompt to query the document
COMPARTMENT_OCID = "ocid1.compartment.oc1..aaaaaaaafm4n4pqttvm6o6p7x2yggwrw2cc2yux23o7mlihzv7lef57m5ksa"

llm = ChatOCIGenAI(
    model_id="meta.llama-3-70b-instruct",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_OCID,
    model_kwargs={"temperature": 0.7, "max_tokens": 400},
)

embed_model = OCIGenAIEmbeddings(
    model_id="cohere.embed-english-v3.0",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_OCID
)

#Set up the template for the questions and context, and instantiate the database retriever object
template = """Answer the question based only on the following context:
            {context} Question: {question} """
prompt = PromptTemplate.from_template(template)

#Retrieval Step 2 - Create retriever without ingesting documents again

vs = OracleVS(embedding_function=embed_model, client=conn23c, table_name="MY_DEMO", distance_strategy=DistanceStrategy.DOT_PRODUCT)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

user_question = ("Tell us about Module 4 of AI Foundations Certification course.")

response = chain.invoke(user_question)

print("User question was ->", user_question)
print("LLM response is ->", response)