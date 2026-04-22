
def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Mock API call to capture a qualified lead.
    In production, this would POST to a CRM like HubSpot or Salesforce.
    """
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"