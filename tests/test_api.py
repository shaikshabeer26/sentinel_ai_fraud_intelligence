import urllib.request
import json

data = {
    "TransactionDT": 86400,
    "TransactionAmt": 750,
    "card1": 10001,
    "card2": 150,
    "card3": 150,
    "card5": 200,
    "addr1": 300,
    "addr2": 87,
    "dist1": 150,
    "ProductCD": "W",
    "card4": "visa",
    "card6": "credit",
    "P_emaildomain": "gmail.com",
    "R_emaildomain": "unknown"
}

json_data = json.dumps(data).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8000/analyze",
    data=json_data,
    headers={
        "Content-Type": "application/json"
    },
    method="POST"
)

try:
    response = urllib.request.urlopen(req)
    print(response.read().decode())

except Exception as e:
    print("Error connecting to API:")
    print(e)
