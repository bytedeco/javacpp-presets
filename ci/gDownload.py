#Taken from answer provided here https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

import os,sys,requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    print 'Running google drive downloader..' 
    if len(sys.argv) == 3:
       file_id = sys.argv[1]
       destination = sys.argv[2]
       download_file_from_google_drive(file_id, destination)
    elif len(sys.argv) == 2:
       file_id = os.environ['GDOWNLOAD_ID']
       destination = sys.argv[1]
       download_file_from_google_drive(file_id, destination)
    else:
       print '2 params needed, file ID and output'
