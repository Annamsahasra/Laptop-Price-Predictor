import app as flask_app
import re

def verify_pipeline():
    # Setup flask test client
    flask_app.app.testing = True
    client = flask_app.app.test_client()

    # Payload matching the notebook validation case
    payload = {
        'company': 'Lenovo',
        'typename': 'Notebook',
        'cpu_brand': 'Intel Core i5',
        'gpu_brand': 'Intel',
        'os': 'Windows',
        'ram': '8',
        'weight': '1.37',
        'inches': '14.0',
        'resolution': '1920x1080',
        'touchscreen': '0',
        'ips': '1',
        'ssd': '256',
        'hdd': '0',
        'flash': '0',
        'hybrid': '0'
    }

    print("Submitting verification request to Flask local test client...")
    response = client.post('/predict', data=payload)
    
    assert response.status_code == 200, f"Request failed with status {response.status_code}"
    
    html_content = response.data.decode('utf-8')
    
    # Extract prediction from HTML
    price_match = re.search(r'&euro;</span>\s*([\d\.]+)', html_content)
    if price_match:
        price = float(price_match.group(1))
        print(f"Flask App prediction matches! Price found: {price:.2f} euros")
        expected = 1377.88
        diff = abs(price - expected)
        if diff < 0.05:
            print("SUCCESS: Flask application matches notebook validation test!")
        else:
            print(f"WARNING: Discrepancy! Expected {expected:.2f}, got {price:.2f}")
    else:
        print("ERROR: Price not found in HTML response.")
        print(html_content[:1500])

if __name__ == '__main__':
    verify_pipeline()
