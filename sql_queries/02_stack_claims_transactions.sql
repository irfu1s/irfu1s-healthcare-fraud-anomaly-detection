CREATE TABLE comb_claims AS

SELECT 
    DESYNPUF_ID AS patient_id,
    CLM_ID AS transaction_id,
    PRVDR_NUM AS provider_id,
    'Inpatient' AS claim_type,
    SUBSTR(CAST(CLM_FROM_DT AS TEXT), 1, 4) AS claim_year,
    CLM_FROM_DT AS service_start,
    CLM_THRU_DT AS service_end,
    CLM_ADMSN_DT AS admission_date,
    CLM_PMT_AMT AS billing_amount,
    CLM_UTLZTN_DAY_CNT AS service_duration,
    CLM_DRG_CD AS treatment_group,
    ADMTNG_ICD9_DGNS_CD AS diagnosis_context,
    ICD9_PRCDR_CD_1 AS primary_procedure,
    HCPCS_CD_1 AS primary_service,
    NCH_BENE_IP_DDCTBL_AMT AS deductible_context,
    NCH_BENE_PTA_COINSRNC_LBLTY_AM AS coinsurance_context
FROM inpatient_claims

UNION ALL

-- 2. Grab all Outpatient Claims (Filling the gaps with NULL)
SELECT 
    DESYNPUF_ID AS patient_id,
    CLM_ID AS transaction_id,
    PRVDR_NUM AS provider_id,
    'Outpatient' AS claim_type,
    SUBSTR(CAST(CLM_FROM_DT AS TEXT), 1, 4) AS claim_year,
    CLM_FROM_DT AS service_start,
    CLM_THRU_DT AS service_end,
    NULL AS admission_date,           -- Outpatients do not get admitted
    CLM_PMT_AMT AS billing_amount,
    NULL AS service_duration,         -- Handled differently in outpatient
    NULL AS treatment_group,          -- No DRG codes for outpatient
    ICD9_DGNS_CD_1 AS diagnosis_context, -- Using primary diagnosis
    ICD9_PRCDR_CD_1 AS primary_procedure,
    HCPCS_CD_1 AS primary_service,
    NULL AS deductible_context,       -- Outpatient uses Part B deductible (kept NULL to align columns)
    NULL AS coinsurance_context       -- Outpatient uses Part B coinsurance
FROM outpatient_claims;