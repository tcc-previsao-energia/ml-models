import requests

def download_file_from_s3(url, local_filename):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Arquivo baixado com sucesso e salvo como {local_filename}!")
        else:
            print(f"Erro ao baixar o arquivo. Status code: {response.status_code}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")



def upload_file_to_s3(local_file_path, s3_file_path, bucket_name):
    try:
        # Constrói o URL de destino no S3
        s3_url = f'https://{bucket_name}.s3.amazonaws.com/{s3_file_path}'

        # Faz o upload do arquivo usando requests
        with open(local_file_path, 'rb') as f:
            response = requests.put(s3_url, data=f)

        if response.status_code == 200:
            print(f'Arquivo {local_file_path} enviado com sucesso para {s3_file_path} no bucket {bucket_name}')
        else:
            print(f'Falha ao enviar arquivo para o S3. Status code: {response.status_code}')

    except Exception as e:
        print(f'Erro ao enviar arquivo para o S3: {e}')