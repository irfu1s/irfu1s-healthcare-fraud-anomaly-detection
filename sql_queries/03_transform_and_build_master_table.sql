CREATE TABLE Healthcare_transactions AS

SELECT 
    -- Transaction Identifiers
    c.transaction_id,
    c.patient_id,
    c.provider_id,
    c.claim_type,
    
    -- Temporal Data (Forced into YYYY-MM-DD format)
    SUBSTR(CAST(c.service_start AS TEXT), 1, 4) || '-' || 
    SUBSTR(CAST(c.service_start AS TEXT), 5, 2) || '-' || 
    SUBSTR(CAST(c.service_start AS TEXT), 7, 2) AS service_start,
    
    SUBSTR(CAST(c.service_end AS TEXT), 1, 4) || '-' || 
    SUBSTR(CAST(c.service_end AS TEXT), 5, 2) || '-' || 
    SUBSTR(CAST(c.service_end AS TEXT), 7, 2) AS service_end,
    
    SUBSTR(CAST(c.admission_date AS TEXT), 1, 4) || '-' || 
    SUBSTR(CAST(c.admission_date AS TEXT), 5, 2) || '-' || 
    SUBSTR(CAST(c.admission_date AS TEXT), 7, 2) AS admission_date,
    
    -- Financial Data (Forced to REAL/Decimal for ML math)
    CAST(c.billing_amount AS REAL) AS billing_amount,
    CAST(c.deductible_context AS REAL) AS deductible_context,
    CAST(c.coinsurance_context AS REAL) AS coinsurance_context,
    
    -- Clinical Data
    CAST(c.service_duration AS INTEGER) AS service_duration,
    c.treatment_group,
    c.diagnosis_context,
    c.primary_procedure,
    c.primary_service,
    
    -- Patient Context Data (Forced into YYYY-MM-DD format)
    SUBSTR(CAST(b.patient_dob AS TEXT), 1, 4) || '-' || 
    SUBSTR(CAST(b.patient_dob AS TEXT), 5, 2) || '-' || 
    SUBSTR(CAST(b.patient_dob AS TEXT), 7, 2) AS patient_dob,
    
    b.gender_context,
    b.demographic_context,
    b.has_diabetes,
    b.has_chf,
    b.has_cancer,
    b.has_copd

FROM comb_claims c
INNER JOIN comb_benf b 
    ON c.patient_id = b.patient_id 
    AND c.claim_year = b.bene_year;