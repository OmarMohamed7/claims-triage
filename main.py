import pandas as pd
from src.agents.intake import extract_claim
from tests.test_intake import mock_claim_submissions

def main():
    data = pd.read_csv("data/raw/claims.csv")

    print(data.dtypes)
    
    # Testing intake agent 
    extracted_claims = []
    
    for i in mock_claim_submissions:
        res = extract_claim(i)
        print(f"Submission ID: {res.submission_id}, Confidence: {res.extraction_confidence}, Missing Fields: {res.missing_fields}")


if __name__ == "__main__":
    main()
