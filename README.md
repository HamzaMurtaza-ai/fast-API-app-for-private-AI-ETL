# fast-API-app-for-private-AI-ETL
 Built whole workflow in Fast API with ollama, langchain and qdrant using bge-m3:latest embedding model .  the  API  endpoint takes google drive file ID and then start its overall process .

 The whole wokrflow application is completely build in python code. NO Automation platform is lniked. 
currently, achieving 2 MB / minute timing for completing whole process from loading to storing , testing using CPU on local machine using local ollama bge-m3:latest embedding model setup and local qdrant vector DB. 


THE WHOLE PROCESS INVOLVES :

1. GDRIVE AUTHENTICATION AND CONNECTION WITH FAST API PYTHON
2. langchain FUNCTIONS TO DOWNLOAD AND EXTRACT CONTENT from files INCLUDING TXT , PDF AND CSV. 
3. CREATION OF CHUNKS AND EMBEDDINGS USING OPEN SOURCE EMBEDDING MODEL USING OLLAMA 
4. STORE OF EMBEDDINGS IN QUDRANT VECTOR DB


   TO RUN THE PLATFORM :
   1. INSTALL PYTHON
   2. CREATE VIRTUAL ENVIRONMENT
   3. ACTIVE THE ENVIRONMENT
   4. RUN THE REQUIREMENTS.TXT FILE AND INSTALL ALL THE PACKAGES.
   5. RUN FAST API


   # ---- FASTAPI ENDPOINT ----
@app.api_route("/process_drive_file/{file_id}"
