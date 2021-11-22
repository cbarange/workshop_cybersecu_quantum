# Workshop Cybersécurité Quantique
> cbarange | 22th November 2021
---

## Génération d'une clé SSH

```bash

pip3 install qiskit
pip3 install cirq

openssl genrsa -out rsa.pem 1024 # Create Private key
openssl rsa -in rsa.pem -outform PEM -pubout -out public.pem # Create Public from Private key 


# Parser https://8gwifi.org/PemParserFunctions.jsp
# Should find prime numbers of modulus


```

http://nombrespremiersliste.free.fr/
https://www.bigprimes.net/archive/prime



Type de clé et taille minimale :
dsa (1024) | ecdsa(256) | ed25519 | ed25519-sk | rsa(1024)


##


cat rsa.pub | cut -d " " -f2 | base64 -d | hexdump -ve '/1 "%02x "' -e '2/8 "\n"'





6767248017843687643214352438904540222331551402044196136561231507660909036269933893129400733068580421834649648248865326760552702714388186006568543091767071


10142789312725007


100711423 * 100711409





7012847*7021951=49243868004497


49243868004497













97 * 71 = 6887
13 * 5 = 65

19 * 41 = 779



2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97