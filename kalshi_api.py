import requests
import datetime
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def load_private_key_from_file(file_path):
    with open(file_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,  # or provide a password if your key is encrypted
            backend=default_backend()
        )
    return private_key

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e:
        raise ValueError("RSA sign PSS failed") from e

def fetch_kalshi_market(api_key, private_key_path, market_path):
    current_time = datetime.datetime.now()
    timestamp_ms = int(current_time.timestamp() * 1000)
    method = "GET"
    base_url = 'https://demo-api.kalshi.co'
    path = market_path
    
    private_key = load_private_key_from_file(private_key_path)
    msg_string = str(timestamp_ms) + method + path
    sig = sign_pss_text(private_key, msg_string)
    
    headers = {
        'KALSHI-ACCESS-KEY': api_key,
        'KALSHI-ACCESS-SIGNATURE': sig,
        'KALSHI-ACCESS-TIMESTAMP': str(timestamp_ms)
    }
    response = requests.get(base_url + path, headers=headers)
    return response.json()
