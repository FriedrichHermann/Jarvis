from pymongo import MongoClient

 

# Create a client instance of MongoClient

client = MongoClient('mongodb+srv://<usr>:<pwd>@outbound-production.h7ybo.mongodb.net/outbound')

 

# Get the database object

# Here name of the database is "sample"
db = client.outbound

 

# Get the collection object

# Here name of the database is "states"

collection  = db.summary

 
lister=[]
# Make a query to list all the documents

for doc in collection.find():

    #Print each document
    lister.append(doc)

l=extractor(lister)