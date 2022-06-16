import base64
from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto import Random

def encryptAES(key, text):
    '''
    Encrypt the given text with the key provided with AES

    Arguments:
    key -- The key given by the user
    text -- The data that needs to by encrypted

    Return:
    The encrypted string
    '''
    key = key.encode()
    text = text.encode()
    key = SHA256.new(key).digest()  # Use SHA-256 over our key to get a proper-sized AES key
    iv = Random.new().read(AES.block_size)  # Generate IV
    encryptor = AES.new(key, AES.MODE_CBC, iv)
    padding = AES.block_size - len(text) % AES.block_size  # calculate needed padding
    text += bytes([padding]) * padding
    data = iv + encryptor.encrypt(text)  # Store the IV at the beginning and encrypt
    return base64.b64encode(data).decode()

def decryptAES(key, text):
    '''
    Decrypt the given text with the key provided with AES

    Arguments:
    key -- The key given by the user
    text -- The data that needs to by decrypted

    Return:
    The decrypted string
    '''
    key = key.encode()
    text = text.encode()
    text = base64.b64decode(text)
    key = SHA256.new(key).digest()  # Use SHA-256 over our key to get a proper-sized AES key
    IV = text[:AES.block_size]  # Extract the IV from the beginning
    decryptor = AES.new(key, AES.MODE_CBC, IV)
    data = decryptor.decrypt(text[AES.block_size:])  # decrypt
    padding = data[-1]  # Pick the padding value from the end; Python 2.x: ord(data[-1])
    if data[-padding:] != bytes([padding]) * padding:
        raise ValueError("Invalid padding...")
    return data[:-padding].decode()  # Remove the padding and decode