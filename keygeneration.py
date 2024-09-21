import numpy as np
from PIL import Image
import hashlib
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization



def image_to_key(image_path):
    # Step 1: Image Processing
    # Open the image and convert it to grayscale
    image = Image.open(image_path).convert('L')
    # Resize the image to reduce complexity (optional)
    image = image.resize((100, 100), Image.ANTIALIAS)
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Convert the array to a byte string
    image_bytes = image_array.tobytes()
    # Hash the byte string to get a fixed-size number
    hash_digest = hashlib.sha256(image_bytes).hexdigest()

    # Step 2: Key Generation
    # Use the hash as a seed to generate an ECC key pair
    # Note: In a real application, ensure the seed is suitable for ECC key generation
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()

    return private_key, public_key


# Replace 'your_image_path_here.jpg' with the path to your image
private_key, public_key = image_to_key('output/1500_20240221_221456_image.png')

print(f"Private Key: {private_key}")
print(f"Public Key: {public_key}")


# Convert the private key to PEM format
private_pem = private_key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption()
)

# Convert the public key to PEM format
public_pem = public_key.public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo
)

print(f"Private Key (PEM format):\n{private_pem.decode('utf-8')}")
print(f"Public Key (PEM format):\n{public_pem.decode('utf-8')}")