CREATE TABLE comb_benf AS

SELECT 
    DESYNPUF_ID AS patient_id, 
    BENE_BIRTH_DT AS patient_dob, 
    BENE_SEX_IDENT_CD AS gender_context, 
    BENE_RACE_CD AS demographic_context, 
    SP_DIABETES AS has_diabetes, 
    SP_CHF AS has_chf, 
    SP_CNCR AS has_cancer, 
    SP_COPD AS has_copd, 
    '2008' AS bene_year 
FROM beneficiary_2008

UNION ALL

SELECT 
    DESYNPUF_ID AS patient_id,
	BENE_BIRTH_DT AS patient_dob, 
	BENE_SEX_IDENT_CD AS gender_context, 
	BENE_RACE_CD AS demographic_context, 
	SP_DIABETES AS has_diabetes, 
	SP_CHF AS has_chf, 
	SP_CNCR AS has_cancer, 
	SP_COPD AS has_copd, 
	'2009' AS bene_year 
FROM beneficiary_2009

UNION ALL

SELECT 
    DESYNPUF_ID AS patient_id,
	BENE_BIRTH_DT AS patient_dob, 
	BENE_SEX_IDENT_CD AS gender_context,
	BENE_RACE_CD AS demographic_context, 
	SP_DIABETES AS has_diabetes, 
	SP_CHF AS has_chf, 
	SP_CNCR AS has_cancer, 
	SP_COPD AS has_copd, 
	'2010' AS bene_year 
FROM beneficiary_2010;