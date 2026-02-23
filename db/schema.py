"""
AML Financial Graph Schema
==========================
Standard financial crime graph model following ACAMS / FATF guidelines.

Node Labels:
  - Customer       : Individual or corporate entity
  - Account        : Bank / wallet account owned by a Customer
  - Transaction    : A financial movement between accounts
  - Device         : Client device used to initiate a transaction
  - IPAddress      : Network origin of a transaction
  - Merchant       : A business receiving payments
  - BeneficiaryAccount : External destination account (for wire transfers)
  - Country        : Jurisdiction node for geographic risk

Relationships:
  - (Customer)-[:OWNS]->(Account)
  - (Account)-[:INITIATED]->(Transaction)
  - (Transaction)-[:CREDITED_TO]->(Account)
  - (Transaction)-[:PAID_TO]->(Merchant)
  - (Transaction)-[:ORIGINATED_FROM]->(Device)
  - (Transaction)-[:SOURCED_FROM]->(IPAddress)
  - (Transaction)-[:SENT_TO_EXTERNAL]->(BeneficiaryAccount)
  - (Customer)-[:RESIDENT_OF]->(Country)
  - (Account)-[:BASED_IN]->(Country)
  - (IPAddress)-[:GEOLOCATED_IN]->(Country)
  - (Customer)-[:LINKED_TO]->(Customer)   [shared device / IP]
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT account_id IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE",
    "CREATE CONSTRAINT transaction_id IF NOT EXISTS FOR (t:Transaction) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT device_id IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT ip_address IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.ip IS UNIQUE",
    "CREATE CONSTRAINT merchant_id IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT beneficiary_id IF NOT EXISTS FOR (b:BeneficiaryAccount) REQUIRE b.id IS UNIQUE",
    "CREATE CONSTRAINT country_code IF NOT EXISTS FOR (c:Country) REQUIRE c.code IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX transaction_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
    "CREATE INDEX transaction_amount IF NOT EXISTS FOR (t:Transaction) ON (t.amount)",
    "CREATE INDEX transaction_status IF NOT EXISTS FOR (t:Transaction) ON (t.status)",
    "CREATE INDEX account_status IF NOT EXISTS FOR (a:Account) ON (a.status)",
    "CREATE INDEX customer_pep IF NOT EXISTS FOR (c:Customer) ON (c.pep_flag)",
    "CREATE INDEX customer_sanctions IF NOT EXISTS FOR (c:Customer) ON (c.sanctions_flag)",
]

# Cypher query: retrieve full transaction context (used by feature extractor)
TRANSACTION_CONTEXT_QUERY = """
MATCH (sender:Account)-[:INITIATED]->(t:Transaction {id: $txn_id})
OPTIONAL MATCH (t)-[:CREDITED_TO]->(receiver:Account)
OPTIONAL MATCH (sender)<-[:OWNS]-(sender_customer:Customer)
OPTIONAL MATCH (receiver)<-[:OWNS]-(receiver_customer:Customer)
OPTIONAL MATCH (t)-[:ORIGINATED_FROM]->(device:Device)
OPTIONAL MATCH (t)-[:SOURCED_FROM]->(ip:IPAddress)
OPTIONAL MATCH (t)-[:PAID_TO]->(merchant:Merchant)
OPTIONAL MATCH (t)-[:SENT_TO_EXTERNAL]->(beneficiary:BeneficiaryAccount)
OPTIONAL MATCH (sender)-[:BASED_IN]->(sender_country:Country)
OPTIONAL MATCH (receiver)-[:BASED_IN]->(receiver_country:Country)
RETURN t, sender, receiver, sender_customer, receiver_customer,
       device, ip, merchant, beneficiary, sender_country, receiver_country
"""

# Velocity: count transactions from sender account in last N hours
SENDER_VELOCITY_QUERY = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(t:Transaction)
WHERE t.timestamp >= $since AND t.id <> $txn_id
RETURN count(t) AS count, sum(t.amount) AS total_amount
"""

# Network depth: how many hops does money travel from this account in 24h
NETWORK_HOP_QUERY = """
MATCH path = (a:Account {id: $account_id})-[:INITIATED|CREDITED_TO*1..5]-(other:Account)
WHERE all(r in relationships(path) WHERE type(r) IN ['INITIATED','CREDITED_TO'])
RETURN count(DISTINCT other) AS connected_accounts, length(path) AS max_hops
ORDER BY max_hops DESC LIMIT 1
"""

# Structuring detection: transactions just below reporting threshold (9000-9999)
STRUCTURING_QUERY = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(t:Transaction)
WHERE t.amount >= 9000 AND t.amount < 10000
  AND t.timestamp >= $since
RETURN count(t) AS structuring_count
"""

# Shared device: how many customers use the same device
SHARED_DEVICE_QUERY = """
MATCH (t:Transaction {id: $txn_id})-[:ORIGINATED_FROM]->(d:Device)
MATCH (other_t:Transaction)-[:ORIGINATED_FROM]->(d)
MATCH (a:Account)-[:INITIATED]->(other_t)
MATCH (c:Customer)-[:OWNS]->(a)
RETURN count(DISTINCT c) AS device_user_count
"""

# Round-tripping: money leaves account and returns within 48h
ROUND_TRIP_QUERY = """
MATCH (a:Account {id: $account_id})-[:INITIATED]->(t_out:Transaction)-[:CREDITED_TO]->(b:Account)
MATCH (b)-[:INITIATED]->(t_in:Transaction)-[:CREDITED_TO]->(a)
WHERE t_out.timestamp >= $since AND t_in.timestamp >= $since
RETURN count(*) AS round_trip_count
"""


def apply_schema(session) -> None:
    for constraint in CONSTRAINTS:
        session.run(constraint)
    for index in INDEXES:
        session.run(index)
