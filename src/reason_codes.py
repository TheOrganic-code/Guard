def get_all_reason_details():
    """
    Returns reference metadata for all possible reason codes.
    """
    return {
        'BOB-RC01': {
            'title': 'High Device Risk Profile',
            'severity': 'CRITICAL',
            'description': 'The device risk score is elevated, indicating a potential emulator, rooting/jailbreaking, or blacklisted device fingerprint.',
            'action': 'Initiate Out-of-Band (OOB) Video KYC or freeze the account recovery.'
        },
        'BOB-RC02': {
            'title': 'High IP Risk Score',
            'severity': 'HIGH',
            'description': 'The transaction IP address is flagged as high-risk, originating from a proxy, VPN, TOR node, or hosting center.',
            'action': 'Request secondary biometric validation or delay the request by 24 hours.'
        },
        'BOB-RC03': {
            'title': 'Repeated Recovery Failures',
            'severity': 'CRITICAL',
            'description': 'There have been 3 or more failed recovery attempts on this account within the last 7 days, indicating potential brute-forcing.',
            'action': 'Temporarily lock account recovery for 48 hours and send alert to registered contact details.'
        },
        'BOB-RC04': {
            'title': 'Suspicious Geodiversity Profile',
            'severity': 'HIGH',
            'description': 'A new device or IP address was used, and a geographic mismatch was detected compared to the last successful login.',
            'action': 'Verify identity via registered Email/SMS secondary backup channels.'
        },
        'BOB-RC05': {
            'title': 'Velocity Velocity Anomaly',
            'severity': 'MEDIUM',
            'description': 'The time between account onboarding and recovery request was exceptionally low (velocity flag triggered).',
            'action': 'Hold recovery attempt for administrative compliance audit.'
        }
    }

def generate_reason_codes(row):
    """
    Generates rule-based reason codes for a given transaction row.
    Returns a list of dictionaries with code details.
    """
    reasons = []
    
    # 1. Device risk
    dev_risk = float(row.get('device_risk_score', 0))
    if dev_risk > 0.65:
        reasons.append({
            'code': 'BOB-RC01',
            'title': 'High Device Risk Profile',
            'severity': 'CRITICAL',
            'value': f'{dev_risk:.2f}',
            'description': f'Device risk score is {dev_risk:.2f} (> 0.65), indicating a potential emulator or jailbroken OS.',
            'action': 'Initiate Out-of-Band (OOB) Video KYC or freeze the account recovery.'
        })
        
    # 2. IP risk
    ip_risk = float(row.get('ip_risk_score', 0))
    if ip_risk > 0.65:
        reasons.append({
            'code': 'BOB-RC02',
            'title': 'High IP Risk Score',
            'severity': 'HIGH',
            'value': f'{ip_risk:.2f}',
            'description': f'IP risk score is {ip_risk:.2f} (> 0.65), indicating a VPN, proxy, or blacklisted network block.',
            'action': 'Request secondary biometric validation or delay the request by 24 hours.'
        })
        
    # 3. Repeated failures
    failures = int(row.get('failed_recovery_attempts_7d', 0))
    if failures >= 3:
        reasons.append({
            'code': 'BOB-RC03',
            'title': 'Repeated Recovery Failures',
            'severity': 'CRITICAL',
            'value': str(failures),
            'description': f'Account has {failures} failed recovery attempts in the last 7 days, showing possible brute-force testing.',
            'action': 'Temporarily lock account recovery for 48 hours and send alert to registered contact details.'
        })
        
    # 4. Geodiversity Profile
    new_device = int(row.get('is_new_device', 0))
    new_ip = int(row.get('is_new_ip', 0))
    geo_mismatch = int(row.get('geo_mismatch_flag', 0))
    if (new_device == 1 or new_ip == 1) and geo_mismatch == 1:
        reasons.append({
            'code': 'BOB-RC04',
            'title': 'Suspicious Geodiversity Profile',
            'severity': 'HIGH',
            'value': f'Device={new_device}, IP={new_ip}, Geo={geo_mismatch}',
            'description': 'A new device/IP was used in combination with a geographic mismatch from previous login history.',
            'action': 'Verify identity via registered Email/SMS secondary backup channels.'
        })
        
    # 5. Speed velocity anomaly
    speed_flag = int(row.get('onboarding_to_recovery_speed_flag', 0))
    if speed_flag == 1:
        reasons.append({
            'code': 'BOB-RC05',
            'title': 'Velocity Velocity Anomaly',
            'severity': 'MEDIUM',
            'value': 'Active',
            'description': 'Extremely short duration between account onboarding and password/credential recovery attempt.',
            'action': 'Hold recovery attempt for administrative compliance audit.'
        })
        
    return reasons
