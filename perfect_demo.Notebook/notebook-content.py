# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "53621640-7d0b-4681-9a90-9036b1e670ea",
# META       "default_lakehouse_name": "LH_Demo",
# META       "default_lakehouse_workspace_id": "f93a16f5-a2b3-432e-bcb3-30877925f57b",
# META       "known_lakehouses": [
# META         {
# META           "id": "53621640-7d0b-4681-9a90-9036b1e670ea"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Fabric Materialized Lake Views → GraphQL — live demo
# 
# Run **top to bottom**. Each cell does one job.
# 
# | Section | Produces |
# |---|---|
# | **Setup** | schemas: bronze, silver, gold, graphql |
# | **Bronze** | raw source tables (CDF on) |
# | **Silver** | cleansed MLVs (1 PySpark + SQL) |
# | **Gold** | star schema (dims + fact_encounter) |
# | **Serving (graphql)** | 3 API objects → ward board · HMO feed · lab alerts |


# MARKDOWN ********************

# ## 1 · Setup

# CELL ********************

# SETUP — create all schemas once (lowercase; MLV schema names can't be all-caps)
for s in ["bronze", "silver", "gold", "graphql"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {s}")
print("schemas ready:", ["bronze", "silver", "gold", "graphql"])


# MARKDOWN ********************

# ## 2 · Bronze — raw sources

# CELL ********************

# ============================================================================
# BRONZE 1/3 — Raw source generator
# patients · departments · providers · diagnoses · encounters · lab_orders
# ----------------------------------------------------------------------------

from pyspark.sql import functions as F

# ---------------------------------------------------------------- config ----
N_PATIENTS     = 3_000
N_PROVIDERS    = 150
N_DEPARTMENTS  = 15
N_DIAGNOSES    = 120
N_ENCOUNTERS   = 25_000
MAX_LABS_PER_ENCOUNTER = 4
BRONZE         = "bronze"
SEED           = 42

# --------------------------------------------------------------- helpers ----
def pick(*vals):
    arr = F.array(*[F.lit(v) for v in vals])
    idx = (F.floor(F.rand(SEED) * F.lit(len(vals))) + F.lit(1)).cast("int")
    return F.element_at(arr, idx)


def maybe_null(col, p=0.03):
    return F.when(F.rand() < F.lit(p), F.lit(None)).otherwise(col)


def write_bronze(df, name):
    (df.write.format("delta").mode("overwrite")
       .option("overwriteSchema", "true")
       .option("delta.enableChangeDataFeed", "true")
       .saveAsTable(f"{BRONZE}.{name}"))
    spark.sql(f"ALTER TABLE {BRONZE}.{name} "
              f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    print(f"  ✓ {BRONZE}.{name:<22} {spark.table(f'{BRONZE}.{name}').count():>8,} rows")


print("Generating bronze layer (hospital encounters)...")

# ------------------------------------------------------------ raw_patients ---
FIRST = ["james", "mary", "ada", "tunde", "chioma", "ibrahim", "grace",
         "emeka", "fatima", "david", "blessing", "samuel", "ngozi", "yusuf"]
LAST  = ["okafor", "bello", "adeyemi", "nwosu", "musa", "eze", "ogunleye",
         "abubakar", "obi", "lawal", "okeke", "danladi"]
CITY  = ["Lagos", "Abuja", "Kano", "Ibadan", "Port Harcourt", "Kaduna", "Benin"]

patients = (
    spark.range(1, N_PATIENTS + 1).withColumnRenamed("id", "n")
    .withColumn("patient_bk", F.concat(F.lit("PAT"),
                                       F.lpad(F.col("n").cast("string"), 6, "0")))
    .withColumn("full_name", F.concat_ws(" ", pick(*FIRST), pick(*LAST)))
    .withColumn("gender", pick("M", "F", "m", "f", "Male", "Female"))
    .withColumn("blood_group", pick("A+", "a+", "O-", "O+", "b+ ", "AB-", "ab+"))
    .withColumn("phone", F.concat(F.lit("+234"),
                                  F.lpad((F.rand() * 1e9).cast("long").cast("string"), 9, "0")))
    .withColumn("city", pick(*CITY))
    .withColumn("insurance_type", pick("NHIS", "nhis", "HMO", "hmo ",
                                       "Self-Pay", "self pay", "SELF-PAY"))
    # date of birth as STRING in two different formats -> silver parses both
    .withColumn("_dob", F.date_sub(F.current_date(), (F.rand() * 30000 + 6000).cast("int")))
    .withColumn("dob",
                F.when(F.rand() < 0.5, F.date_format("_dob", "yyyy-MM-dd"))
                 .otherwise(F.date_format("_dob", "dd/MM/yyyy")))
    .withColumn("city", maybe_null(F.col("city"), 0.02))
    .drop("n", "_dob")
)
# ~5% duplicate patient rows with casing variation (same business key)
dupes = (patients.sample(0.05, seed=SEED)
         .withColumn("full_name", F.upper("full_name")))
write_bronze(patients.unionByName(dupes), "raw_patients")

# ---------------------------------------------------------- raw_departments ---
departments = (
    spark.range(1, N_DEPARTMENTS + 1).withColumnRenamed("id", "n")
    .withColumn("department_bk", F.concat(F.lit("DEP"),
                                          F.lpad(F.col("n").cast("string"), 3, "0")))
    .withColumn("department_name", pick("Cardiology", "Pediatrics", "Oncology",
                                        "Orthopedics", "Neurology", "Radiology",
                                        "General Medicine", "Emergency", "Maternity"))
    .withColumn("ward_type", pick("Inpatient", "inpatient", "OUTPATIENT",
                                  "Outpatient", "ICU", "icu"))
    .withColumn("building", pick("Block A", "Block B", "Annex", "block a"))
    .drop("n")
)
write_bronze(departments, "raw_departments")

# ------------------------------------------------------------ raw_providers ---
providers = (
    spark.range(1, N_PROVIDERS + 1).withColumnRenamed("id", "n")
    .withColumn("provider_bk", F.concat(F.lit("DOC"),
                                        F.lpad(F.col("n").cast("string"), 4, "0")))
    # inconsistent "Dr" prefixing -> silver strips it
    .withColumn("provider_name",
                F.concat(pick("Dr ", "Dr. ", "DR ", ""),
                         pick(*FIRST), F.lit(" "), pick(*LAST)))
    .withColumn("specialty", pick("Cardiologist", "cardiologist", "PEDIATRICIAN",
                                  "Oncologist", "Neurologist", "general physician",
                                  "Radiologist", "Surgeon"))
    .withColumn("department_bk", F.concat(F.lit("DEP"),
                F.lpad(((F.rand(SEED) * N_DEPARTMENTS).cast("int") + 1).cast("string"), 3, "0")))
    .drop("n")
)
write_bronze(providers, "raw_providers")

# ------------------------------------------------------------ raw_diagnoses ---
DIAGNOSES = [
    "Malaria", "Typhoid Fever", "Type 2 Diabetes Mellitus", "Essential Hypertension",
    "Pneumonia", "Acute Gastroenteritis", "Urinary Tract Infection", "Bronchial Asthma",
    "Sickle Cell Crisis", "Peptic Ulcer Disease", "Lower Respiratory Tract Infection",
    "Congestive Heart Failure", "Ischemic Heart Disease", "Chronic Kidney Disease",
    "Acute Appendicitis", "Femur Fracture", "Road Traffic Injury", "Cellulitis",
    "Pulmonary Tuberculosis", "Hepatitis B", "Iron-Deficiency Anaemia", "Preeclampsia",
    "Postpartum Haemorrhage", "Gestational Diabetes", "Bacterial Meningitis", "Sepsis",
    "Ischemic Stroke", "Migraine", "Epilepsy", "Osteoarthritis", "Rheumatoid Arthritis",
    "Dengue Fever", "Acute Bronchitis", "COPD Exacerbation", "Diabetic Ketoacidosis",
    "Hypertensive Crisis", "Pyelonephritis", "Otitis Media", "Acute Tonsillitis",
    "Lumbar Disc Herniation", "Deep Vein Thrombosis", "Pulmonary Embolism",
    "Acute Cholecystitis", "Acute Pancreatitis", "Atopic Dermatitis", "Hypothyroidism",
    "Lobar Pneumonia", "Cholera",
]
_dx_arr = F.array(*[F.lit(d) for d in DIAGNOSES])
diagnoses = (
    spark.range(1, N_DIAGNOSES + 1).withColumnRenamed("id", "n")
    .withColumn("diagnosis_bk", F.concat(F.lit("DX-"),
                                         F.lpad(F.col("n").cast("string"), 4, "0")))
    # real condition name, deterministic by code number
    .withColumn("diagnosis_desc",
                F.element_at(_dx_arr,
                             (F.pmod(F.col("n").cast("int"), F.lit(len(DIAGNOSES))) + F.lit(1)).cast("int")))
    .withColumn("diagnosis_category", pick("Infectious", "infectious", "Chronic",
                                           "CHRONIC", "Injury", "injury ",
                                           "Maternal", "Cardiac"))
    .drop("n")
)
write_bronze(diagnoses, "raw_diagnoses")

# ----------------------------------------------------------- raw_encounters ---
# This is the FACT grain. admit/discharge as STRING timestamps; ₦-string money.
encounters = (
    spark.range(1, N_ENCOUNTERS + 1).withColumnRenamed("id", "n")
    .withColumn("encounter_bk", F.concat(F.lit("ENC"),
                                         F.lpad(F.col("n").cast("string"), 8, "0")))
    .withColumn("patient_bk", F.concat(F.lit("PAT"),
                F.lpad(((F.rand(SEED) * N_PATIENTS).cast("int") + 1).cast("string"), 6, "0")))
    .withColumn("provider_bk", F.concat(F.lit("DOC"),
                F.lpad(((F.rand() * N_PROVIDERS).cast("int") + 1).cast("string"), 4, "0")))
    .withColumn("department_bk", F.concat(F.lit("DEP"),
                F.lpad(((F.rand() * N_DEPARTMENTS).cast("int") + 1).cast("string"), 3, "0")))
    .withColumn("diagnosis_bk", F.concat(F.lit("DX-"),
                F.lpad(((F.rand() * N_DIAGNOSES).cast("int") + 1).cast("string"), 4, "0")))
    # admit in last ~2 years; discharge = admit + 0..14 days (some still admitted)
    .withColumn("_admit", F.from_unixtime(
        F.unix_timestamp(F.current_timestamp()) - (F.rand() * 60 * 60 * 24 * 700).cast("long")))
    .withColumn("_los_days", (F.rand() * 14).cast("int"))
    .withColumn("admit_ts", F.date_format("_admit", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("discharge_ts",
                F.when(F.rand() < 0.08, F.lit(None))   # still admitted -> null
                 .otherwise(F.date_format(
                     F.expr("_admit + make_interval(0, 0, 0, _los_days, 0, 0, 0)"),
                     "yyyy-MM-dd HH:mm:ss")))
    .withColumn("encounter_type", pick("Inpatient", "inpatient", "Outpatient",
                                       "OUTPATIENT", "Emergency", "emergency"))
    .withColumn("status", pick("Discharged", "discharged", "Admitted",
                               "ADMITTED", "Cancelled"))
    # billed ₦-string; inject ~1.5% negative (bad data) for the DQ drop demo
    .withColumn("_billed", F.round(F.rand(SEED) * F.lit(450000) + F.lit(5000), 2))
    .withColumn("_billed", F.when(F.rand() < 0.015, -F.col("_billed")).otherwise(F.col("_billed")))
    .withColumn("billed_amount_raw", F.concat(F.lit("₦"), F.format_number(F.col("_billed"), 2)))
    .withColumn("amount_paid_raw",
                F.concat(F.lit("₦"),
                         F.format_number(F.abs(F.col("_billed")) * F.rand(), 2)))
    .select("encounter_bk", "patient_bk", "provider_bk", "department_bk",
            "diagnosis_bk", "admit_ts", "discharge_ts", "encounter_type",
            "status", "billed_amount_raw", "amount_paid_raw")
)
write_bronze(encounters, "raw_encounters")

# ---------------------------------------------------------- raw_lab_orders ---
# Line grain under encounter; aggregated up to encounter grain in the gold fact.
n_labs_df = (encounters.select("encounter_bk", "admit_ts")
             .withColumn("n_labs",
                         (F.floor(F.rand(SEED) * MAX_LABS_PER_ENCOUNTER) + 1).cast("int")))
lab_orders = (
    n_labs_df
    .withColumn("seq", F.explode(F.expr("sequence(1, n_labs)")))
    .withColumn("lab_order_bk",
                F.concat(F.col("encounter_bk"), F.lit("-LAB"),
                         F.lpad(F.col("seq").cast("string"), 2, "0")))
    .withColumn("test_name", pick("CBC", "Malaria RDT", "Lipid Panel", "LFT",
                                  "Urinalysis", "Blood Glucose", "X-Ray", "MRI"))
    .withColumn("_cost", F.round(F.rand() * F.lit(40000) + F.lit(1500), 2))
    .withColumn("_cost", F.when(F.rand() < 0.01, -F.col("_cost")).otherwise(F.col("_cost")))
    .withColumn("test_cost_raw", F.concat(F.lit("₦"), F.format_number(F.col("_cost"), 2)))
    .withColumn("result_flag", maybe_null(pick("Normal", "normal", "Abnormal",
                                               "ABNORMAL"), 0.05))
    # result timestamp baked in at source: admit + 1..72h, null if not yet resulted
    .withColumn("_admit_ts", F.to_timestamp("admit_ts", "yyyy-MM-dd HH:mm:ss"))
    .withColumn("_offset_h", (F.rand() * 71 + 1).cast("int"))
    .withColumn("resulted_ts",
                F.when(F.col("result_flag").isNull(), F.lit(None))
                 .otherwise(F.date_format(
                     F.expr("_admit_ts + make_interval(0,0,0,0,_offset_h,0,0)"),
                     "yyyy-MM-dd HH:mm:ss")))
    .select("lab_order_bk", "encounter_bk", "test_name", "test_cost_raw",
            "result_flag", "resulted_ts")
)
write_bronze(lab_orders, "raw_lab_orders")

print("\nBronze ready. CDF enabled on all source tables.")
print("Tables:", [t.name for t in spark.catalog.listTables(BRONZE)])


# CELL ********************

# ============================================================================
# BRONZE 2/3 — Bed assignments (ADT / bed-management feed)
# ----------------------------------------------------------------------------
# Stands in for an ADT / bed-management feed. Reads the EXISTING
# bronze.raw_encounters and derives one bed assignment per encounter so that:
#     released_ts IS NULL  <=>  discharge_ts IS NULL  <=>  currently admitted
# Ward number is derived from the encounter's department (ward = specialty).
# Run once; nothing upstream changes.
# ============================================================================

from pyspark.sql import functions as F

BRONZE = "bronze"


def pick(*vals):
    arr = F.array(*[F.lit(v) for v in vals])
    idx = (F.floor(F.rand() * F.lit(len(vals))) + F.lit(1)).cast("int")
    return F.element_at(arr, idx)


enc = spark.table(f"{BRONZE}.raw_encounters")

bed_assignments = (
    enc.select("encounter_bk", "department_bk", "admit_ts", "discharge_ts")
    # one bed per encounter (transfers would simply be extra rows here)
    .withColumn("bed_assignment_bk", F.concat(F.col("encounter_bk"), F.lit("-BED01")))
    # ward derived from department number, with light casing/space dirtiness
    .withColumn("ward_no",
                F.concat(pick("W", "w", "W "), F.substring("department_bk", 4, 3)))
    .withColumn("bed_no",
                F.concat(pick("B", "b"),
                         F.lpad(((F.rand() * 40).cast("int") + 1).cast("string"), 2, "0")))
    .withColumn("assigned_ts", F.col("admit_ts"))
    .withColumn("released_ts", F.col("discharge_ts"))   # null => still in the bed
    .select("bed_assignment_bk", "encounter_bk", "ward_no", "bed_no",
            "assigned_ts", "released_ts")
)

(bed_assignments.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")     # source for the silver MLV
    .saveAsTable(f"{BRONZE}.raw_bed_assignments"))
spark.sql(f"ALTER TABLE {BRONZE}.raw_bed_assignments "
          f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

total = spark.table(f"{BRONZE}.raw_bed_assignments").count()
active = spark.table(f"{BRONZE}.raw_bed_assignments").filter("released_ts IS NULL").count()
print(f"  ✓ {BRONZE}.raw_bed_assignments  {total:,} rows  ({active:,} currently active)")


# CELL ********************

# ============================================================================
# BRONZE 3/3 — Patient coverage (insurance enrollment feed)
# ----------------------------------------------------------------------------
# Stands in for a coverage/enrollment feed (separate system from the patient
# master). Reads existing bronze.raw_patients, dedupes, and assigns each
# patient an HMO_CODE *deterministically* by hashing the patient key, so the
# insurer-to-patient mapping is stable every run (no random reassignment).
# Run once; nothing upstream changes.
# ============================================================================

from pyspark.sql import functions as F

BRONZE = "bronze"
HMOS = ["HMO-HYGEIA", "HMO-AXA", "HMO-AVON", "HMO-RELIANCE"]   # the insurers

pat = (spark.table(f"{BRONZE}.raw_patients")
       .select("patient_bk", "insurance_type")
       .dropDuplicates(["patient_bk"]))                        # raw has dupes

ins = F.lower(F.trim("insurance_type"))
payer_type = (F.when(ins.contains("nhis"), F.lit("NHIS"))
               .when(ins.contains("hmo"), F.lit("HMO"))
               .otherwise(F.lit("SELF_PAY")))

# deterministic insurer: pmod(hash(key), N) -> stable bucket per patient
hmo_pick = F.element_at(
    F.array(*[F.lit(h) for h in HMOS]),
    (F.pmod(F.xxhash64("patient_bk"), F.lit(len(HMOS))) + F.lit(1)).cast("int"))

coverage = (
    pat
    .withColumn("_payer", payer_type)
    .withColumn("hmo_code",
                F.when(F.col("_payer") == "HMO", hmo_pick)
                 .when(F.col("_payer") == "NHIS", F.lit("NHIS"))
                 .otherwise(F.lit("SELF-PAY")))
    .withColumn("coverage_bk", F.concat(F.lit("COV"), F.substring("patient_bk", 4, 6)))
    .withColumn("policy_no", F.concat(F.lit("POL-"), F.substring("patient_bk", 4, 6)))
    .withColumn("valid_from", F.date_format(
        F.date_sub(F.current_date(), (F.rand() * 1200 + 200).cast("int")), "yyyy-MM-dd"))
    # light casing dirtiness so the silver MLV still has something to clean
    .withColumn("payer_type",
                F.when(F.rand() < 0.3, F.lower(F.col("_payer"))).otherwise(F.col("_payer")))
    .select("coverage_bk", "patient_bk", "payer_type",
            "hmo_code", "policy_no", "valid_from")
)

(coverage.write.format("delta").mode("overwrite")
    .option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(f"{BRONZE}.raw_patient_coverage"))
spark.sql(f"ALTER TABLE {BRONZE}.raw_patient_coverage "
          f"SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

print(f"  ✓ {BRONZE}.raw_patient_coverage  "
      f"{spark.table(f'{BRONZE}.raw_patient_coverage').count():,} rows")
display(spark.sql(f"""
    SELECT hmo_code, count(*) AS patients
    FROM {BRONZE}.raw_patient_coverage
    GROUP BY hmo_code ORDER BY hmo_code
"""))

# MARKDOWN ********************

# ## 3 · Silver — cleansed MLVs

# CELL ********************

# SILVER 1/4 — patient_silver  [config + UDFs]
# The one silver view that justifies PySpark: dateutil parses messy DOB
# formats and a UDF normalizes phones. This cell MUST sit ABOVE the @fmlv
# cell. PySpark-authored MLVs full-refresh — fine, it only feeds dim_patient.
import re
import fmlv
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DateType
from pyspark.sql.window import Window
from dateutil import parser as dateparser   # ships with the Fabric runtime

LH = "LH_Demo"   # <-- set to your attached lakehouse name


@F.udf(returnType=DateType())
def parse_any_date(s):
    """Parse messy DOB strings (ISO, dd/MM/yyyy, ...). dayfirst=True; None on garbage."""
    if s is None:
        return None
    try:
        return dateparser.parse(s.strip(), dayfirst=True).date()
    except Exception:
        return None


@F.udf(returnType=StringType())
def normalize_phone(s):
    """Strip non-digits, drop leading country code / zeros, re-stamp +234."""
    if s is None:
        return None
    digits = re.sub(r"\D", "", s)
    if digits.startswith("234"):
        digits = digits[3:]
    digits = digits.lstrip("0")
    return ("+234" + digits) if digits else None


# CELL ********************

# SILVER 1/4 — patient_silver  [MLV definition]  (one @fmlv decorator per cell)
@fmlv.materialized_lake_view(
    name=f"{LH}.silver.patient_silver",
    comment="Patient master (PySpark): dateutil parsing + phone UDF + dedup",
    replace=True,
)
@fmlv.check("valid_patient_bk", "PATIENT_BK IS NOT NULL", "drop")
def patient_silver():
    raw = spark.read.table("bronze.raw_patients")
    dedup = Window.partitionBy("patient_bk").orderBy(F.col("dob").asc_nulls_last())
    g = F.lower(F.trim("gender"))
    return (
        raw
        .withColumn("FULL_NAME", F.initcap(F.trim("full_name")))
        .withColumn("GENDER",
                    F.when(g.isin("m", "male"), F.lit("Male"))
                     .when(g.isin("f", "female"), F.lit("Female"))
                     .otherwise(F.lit("Unknown")))
        .withColumn("BLOOD_GROUP", F.upper(F.trim("blood_group")))
        .withColumn("CITY", F.coalesce(F.initcap(F.trim("city")), F.lit("Unknown")))
        .withColumn("INSURANCE_TYPE",
                    F.when(F.lower(F.trim("insurance_type")).contains("nhis"), F.lit("NHIS"))
                     .when(F.lower(F.trim("insurance_type")).contains("hmo"), F.lit("HMO"))
                     .otherwise(F.lit("Self-Pay")))
        .withColumn("DATE_OF_BIRTH", parse_any_date(F.col("dob")))        # external lib
        .withColumn("PHONE", normalize_phone(F.col("phone")))            # python UDF
        .withColumn("AGE", F.floor(F.months_between(F.current_date(),
                                                    F.col("DATE_OF_BIRTH")) / 12).cast("int"))
        .withColumn("_rn", F.row_number().over(dedup)).filter("_rn = 1")  # dedupe
        .select(F.col("patient_bk").alias("PATIENT_BK"),
                "FULL_NAME", "GENDER", "BLOOD_GROUP", "PHONE", "CITY",
                "INSURANCE_TYPE", "DATE_OF_BIRTH", "AGE")
    )


# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- SILVER 2/4 — SQL MLVs: department · provider · diagnosis · lab_order · encounter
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- Run in a Spark-SQL notebook (or Lakehouse > Materialized lake views >
# MAGIC -- New > Create with Spark SQL). These are plain projection / cast / regex
# MAGIC -- transforms, so they stay OPTIMAL/INCREMENTAL-refresh eligible (CDF is on by
# MAGIC -- default for SQL-authored MLVs, and bronze sources have CDF enabled).
# MAGIC --
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- silver.department_silver
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.department_silver
# MAGIC COMMENT "Department dimension cleansed"
# MAGIC AS
# MAGIC SELECT
# MAGIC     department_bk                                       AS DEPARTMENT_BK,
# MAGIC     initcap(trim(department_name))                      AS DEPARTMENT_NAME,
# MAGIC     CASE WHEN lower(trim(ward_type)) = 'icu' THEN 'ICU'
# MAGIC          ELSE initcap(trim(ward_type)) END              AS WARD_TYPE,
# MAGIC     initcap(trim(building))                             AS BUILDING
# MAGIC FROM bronze.raw_departments;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- silver.provider_silver  (strip 'Dr'/'Dr.'/'DR' prefixes)
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.provider_silver
# MAGIC COMMENT "Providers cleansed; doctor-title prefixes removed"
# MAGIC AS
# MAGIC SELECT
# MAGIC     provider_bk                                         AS PROVIDER_BK,
# MAGIC     initcap(trim(regexp_replace(provider_name, '(?i)^dr[. ]+', ''))) AS PROVIDER_NAME,
# MAGIC     initcap(trim(specialty))                            AS SPECIALTY,
# MAGIC     department_bk                                       AS DEPARTMENT_BK
# MAGIC FROM bronze.raw_providers;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- silver.diagnosis_silver
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.diagnosis_silver
# MAGIC COMMENT "Diagnosis reference cleansed"
# MAGIC AS
# MAGIC SELECT
# MAGIC     diagnosis_bk                                        AS DIAGNOSIS_BK,
# MAGIC     trim(diagnosis_desc)                                AS DIAGNOSIS_DESC,
# MAGIC     initcap(trim(diagnosis_category))                   AS DIAGNOSIS_CATEGORY
# MAGIC FROM bronze.raw_diagnoses;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- silver.lab_order_silver  (parse ₦-cost; DQ drops the injected negatives)
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.lab_order_silver
# MAGIC (
# MAGIC     CONSTRAINT non_negative_cost CHECK (TEST_COST >= 0) ON MISMATCH DROP
# MAGIC )
# MAGIC COMMENT "Lab orders (line grain) cleansed; non-negative cost; result timestamp"
# MAGIC AS
# MAGIC SELECT
# MAGIC     lab_order_bk                                        AS LAB_ORDER_BK,
# MAGIC     encounter_bk                                        AS ENCOUNTER_BK,
# MAGIC     test_name                                           AS TEST_NAME,
# MAGIC     CAST(regexp_replace(test_cost_raw, '[^0-9.-]', '') AS DECIMAL(18,2)) AS TEST_COST,
# MAGIC     coalesce(initcap(trim(result_flag)), 'Unknown')     AS RESULT_FLAG,
# MAGIC     to_timestamp(resulted_ts, 'yyyy-MM-dd HH:mm:ss')    AS RESULTED_TS
# MAGIC FROM bronze.raw_lab_orders;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- silver.encounter_silver  (parse timestamps + money; derive LOS + year)
# MAGIC --   Partitioned by ADMIT_YEAR (coarse grain -> few, healthy partitions).
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.encounter_silver
# MAGIC PARTITIONED BY (ADMIT_YEAR)
# MAGIC COMMENT "Encounter header cleansed; timestamps + amounts parsed, LOS derived"
# MAGIC AS
# MAGIC SELECT
# MAGIC     encounter_bk                                        AS ENCOUNTER_BK,
# MAGIC     patient_bk                                          AS PATIENT_BK,
# MAGIC     provider_bk                                         AS PROVIDER_BK,
# MAGIC     department_bk                                       AS DEPARTMENT_BK,
# MAGIC     diagnosis_bk                                        AS DIAGNOSIS_BK,
# MAGIC     to_timestamp(admit_ts,     'yyyy-MM-dd HH:mm:ss')   AS ADMIT_TS,
# MAGIC     to_timestamp(discharge_ts, 'yyyy-MM-dd HH:mm:ss')   AS DISCHARGE_TS,
# MAGIC     to_date(to_timestamp(admit_ts, 'yyyy-MM-dd HH:mm:ss'))               AS ADMIT_DATE,
# MAGIC     datediff(to_timestamp(discharge_ts, 'yyyy-MM-dd HH:mm:ss'),
# MAGIC              to_timestamp(admit_ts,     'yyyy-MM-dd HH:mm:ss'))          AS LENGTH_OF_STAY_DAYS,
# MAGIC     initcap(trim(encounter_type))                       AS ENCOUNTER_TYPE,
# MAGIC     initcap(trim(status))                               AS STATUS,
# MAGIC     CAST(regexp_replace(billed_amount_raw, '[^0-9.-]', '') AS DECIMAL(18,2)) AS BILLED_AMOUNT,
# MAGIC     CAST(regexp_replace(amount_paid_raw,   '[^0-9.-]', '') AS DECIMAL(18,2)) AS AMOUNT_PAID,
# MAGIC     CAST(regexp_replace(billed_amount_raw, '[^0-9.-]', '') AS DECIMAL(18,2))
# MAGIC         - CAST(regexp_replace(amount_paid_raw, '[^0-9.-]', '') AS DECIMAL(18,2)) AS OUTSTANDING_AMOUNT,
# MAGIC     year(to_timestamp(admit_ts, 'yyyy-MM-dd HH:mm:ss')) AS ADMIT_YEAR
# MAGIC FROM bronze.raw_encounters;


# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- SILVER 3/4 — bed_assignment_silver
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- Cleans raw_bed_assignments and exposes IS_ACTIVE. Plain casts/casing, so it
# MAGIC -- stays incremental-refresh eligible. Add this alongside the other SQL silver
# MAGIC -- MLVs (02b). Run in a Spark-SQL notebook.
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.bed_assignment_silver
# MAGIC (
# MAGIC     CONSTRAINT valid_assignment CHECK (ENCOUNTER_BK IS NOT NULL) ON MISMATCH DROP
# MAGIC )
# MAGIC COMMENT "Bed assignments cleansed; IS_ACTIVE flags the patient's current bed"
# MAGIC AS
# MAGIC SELECT
# MAGIC     bed_assignment_bk                                   AS BED_ASSIGNMENT_BK,
# MAGIC     encounter_bk                                        AS ENCOUNTER_BK,
# MAGIC     upper(trim(ward_no))                                AS WARD_NO,
# MAGIC     upper(trim(bed_no))                                 AS BED_NO,
# MAGIC     to_timestamp(assigned_ts, 'yyyy-MM-dd HH:mm:ss')    AS ASSIGNED_TS,
# MAGIC     to_timestamp(released_ts, 'yyyy-MM-dd HH:mm:ss')    AS RELEASED_TS,
# MAGIC     (released_ts IS NULL)                               AS IS_ACTIVE
# MAGIC FROM bronze.raw_bed_assignments;


# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- SILVER 4/4 — patient_coverage_silver
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- Cleans the enrollment feed and exposes HMO_CODE + IS_HMO. Plain casts/casing
# MAGIC -- so it stays incremental-refresh eligible. Run in a Spark-SQL notebook.
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC 
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW silver.patient_coverage_silver
# MAGIC (
# MAGIC     CONSTRAINT valid_coverage CHECK (PATIENT_BK IS NOT NULL) ON MISMATCH DROP
# MAGIC )
# MAGIC COMMENT "Patient insurance coverage; HMO_CODE is the per-insurer scoping key"
# MAGIC AS
# MAGIC SELECT
# MAGIC     coverage_bk                                 AS COVERAGE_BK,
# MAGIC     patient_bk                                  AS PATIENT_BK,
# MAGIC     upper(trim(payer_type))                     AS PAYER_TYPE,
# MAGIC     upper(trim(hmo_code))                        AS HMO_CODE,
# MAGIC     policy_no                                   AS POLICY_NO,
# MAGIC     to_date(valid_from, 'yyyy-MM-dd')            AS VALID_FROM,
# MAGIC     (upper(trim(payer_type)) = 'HMO')           AS IS_HMO
# MAGIC FROM bronze.raw_patient_coverage;

# MARKDOWN ********************

# ## 4 · Gold — star schema

# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- GOLD — star schema (dims + fact_encounter)
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- Run in a Spark-SQL notebook after bronze (01), patient PySpark (02a), and
# MAGIC -- silver SQL (02b). Fabric infers the silver -> gold lineage automatically.
# MAGIC --
# MAGIC -- Refresh behaviour (the talking point):
# MAGIC --   * dim_patient sources the PySpark patient_silver (full-refresh overwrite)
# MAGIC --     -> dim_patient also full-refreshes. Tiny table, negligible cost.
# MAGIC --   * Every other dim + fact_encounter source SQL silver MLVs (append-only,
# MAGIC --     CDF on) and use only incremental-eligible constructs (CTE, GROUP BY
# MAGIC --     aggregation, LEFT JOIN) -> OPTIMAL/INCREMENTAL refresh.
# MAGIC --   Enable it under Lakehouse > Materialized lake views > Manage > Optimal
# MAGIC --   refresh, then schedule from the Schedules pane.
# MAGIC --
# MAGIC -- Surrogate keys: xxhash64(business_key) — deterministic + refresh-stable.
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- DIM_PATIENT  (sources the PySpark MLV -> full refresh)
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW gold.dim_patient
# MAGIC (
# MAGIC     CONSTRAINT valid_patient CHECK (PATIENT_BK IS NOT NULL) ON MISMATCH DROP
# MAGIC )
# MAGIC COMMENT "Patient dimension (SCD1)"
# MAGIC AS
# MAGIC SELECT
# MAGIC     xxhash64(PATIENT_BK)            AS PATIENT_SK,
# MAGIC     PATIENT_BK, FULL_NAME, GENDER, BLOOD_GROUP, PHONE, CITY,
# MAGIC     INSURANCE_TYPE, DATE_OF_BIRTH, AGE
# MAGIC FROM silver.patient_silver;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- DIM_PROVIDER  (with department roll-up)
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW gold.dim_provider
# MAGIC COMMENT "Provider dimension with department context"
# MAGIC AS
# MAGIC SELECT
# MAGIC     xxhash64(p.PROVIDER_BK)         AS PROVIDER_SK,
# MAGIC     p.PROVIDER_BK, p.PROVIDER_NAME, p.SPECIALTY,
# MAGIC     p.DEPARTMENT_BK, d.DEPARTMENT_NAME, d.WARD_TYPE
# MAGIC FROM silver.provider_silver p
# MAGIC LEFT JOIN silver.department_silver d ON p.DEPARTMENT_BK = d.DEPARTMENT_BK;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- DIM_DEPARTMENT
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW gold.dim_department
# MAGIC COMMENT "Department dimension"
# MAGIC AS
# MAGIC SELECT
# MAGIC     xxhash64(DEPARTMENT_BK)         AS DEPARTMENT_SK,
# MAGIC     DEPARTMENT_BK, DEPARTMENT_NAME, WARD_TYPE, BUILDING
# MAGIC FROM silver.department_silver;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- DIM_DIAGNOSIS
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW gold.dim_diagnosis
# MAGIC COMMENT "Diagnosis dimension"
# MAGIC AS
# MAGIC SELECT
# MAGIC     xxhash64(DIAGNOSIS_BK)          AS DIAGNOSIS_SK,
# MAGIC     DIAGNOSIS_BK, DIAGNOSIS_DESC, DIAGNOSIS_CATEGORY
# MAGIC FROM silver.diagnosis_silver;
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- DIM_DATE  (generated; no Delta source -> always full refresh, tiny)
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW gold.dim_date
# MAGIC COMMENT "Conformed calendar dimension"
# MAGIC AS
# MAGIC SELECT
# MAGIC     CAST(date_format(d, 'yyyyMMdd') AS INT)  AS DATE_SK,
# MAGIC     d                                        AS DATE_VALUE,
# MAGIC     year(d)                                  AS YEAR,
# MAGIC     quarter(d)                               AS QUARTER,
# MAGIC     month(d)                                 AS MONTH,
# MAGIC     date_format(d, 'MMMM')                   AS MONTH_NAME,
# MAGIC     day(d)                                   AS DAY_OF_MONTH,
# MAGIC     date_format(d, 'EEEE')                   AS DAY_NAME,
# MAGIC     weekofyear(d)                            AS WEEK_OF_YEAR,
# MAGIC     CASE WHEN date_format(d, 'EEEE') IN ('Saturday','Sunday')
# MAGIC          THEN true ELSE false END            AS IS_WEEKEND
# MAGIC FROM (
# MAGIC     SELECT explode(sequence(to_date('2023-01-01'),
# MAGIC                             to_date('2026-12-31'),
# MAGIC                             interval 1 day)) AS d
# MAGIC );
# MAGIC 
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- FACT_ENCOUNTER  (grain = one encounter)
# MAGIC --   CTE aggregation + LEFT JOIN are both incremental-eligible.
# MAGIC --   DQ constraints drop the bad rows injected in bronze.
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW gold.fact_encounter
# MAGIC (
# MAGIC     CONSTRAINT non_negative_billed CHECK (BILLED_AMOUNT >= 0)      ON MISMATCH DROP,
# MAGIC     CONSTRAINT non_negative_los    CHECK (LENGTH_OF_STAY_DAYS >= 0) ON MISMATCH DROP,
# MAGIC     CONSTRAINT not_cancelled       CHECK (STATUS <> 'Cancelled')    ON MISMATCH DROP
# MAGIC )
# MAGIC PARTITIONED BY (ADMIT_YEAR)
# MAGIC COMMENT "Encounter fact conformed to gold dimensions"
# MAGIC AS
# MAGIC WITH labs AS (
# MAGIC     SELECT
# MAGIC         ENCOUNTER_BK,
# MAGIC         CAST(count(*) AS INT)              AS NUM_LAB_TESTS,
# MAGIC         CAST(sum(TEST_COST) AS DECIMAL(18,2)) AS TOTAL_LAB_COST
# MAGIC     FROM silver.lab_order_silver
# MAGIC     GROUP BY ENCOUNTER_BK
# MAGIC )
# MAGIC SELECT
# MAGIC     xxhash64(e.ENCOUNTER_BK)                  AS ENCOUNTER_SK,
# MAGIC     e.ENCOUNTER_BK,
# MAGIC     -- foreign keys (recomputed deterministically -> match the dims)
# MAGIC     xxhash64(e.PATIENT_BK)                    AS PATIENT_SK,
# MAGIC     xxhash64(e.PROVIDER_BK)                   AS PROVIDER_SK,
# MAGIC     xxhash64(e.DEPARTMENT_BK)                 AS DEPARTMENT_SK,
# MAGIC     xxhash64(e.DIAGNOSIS_BK)                  AS DIAGNOSIS_SK,
# MAGIC     CAST(date_format(e.ADMIT_DATE, 'yyyyMMdd') AS INT) AS ADMIT_DATE_SK,
# MAGIC     e.ADMIT_DATE,
# MAGIC     -- degenerate / context
# MAGIC     e.ENCOUNTER_TYPE,
# MAGIC     e.STATUS,
# MAGIC     -- measures
# MAGIC     e.LENGTH_OF_STAY_DAYS,
# MAGIC     coalesce(l.NUM_LAB_TESTS, 0)              AS NUM_LAB_TESTS,
# MAGIC     coalesce(l.TOTAL_LAB_COST, 0)            AS TOTAL_LAB_COST,
# MAGIC     e.BILLED_AMOUNT,
# MAGIC     e.AMOUNT_PAID,
# MAGIC     e.OUTSTANDING_AMOUNT,
# MAGIC     e.ADMIT_YEAR
# MAGIC FROM silver.encounter_silver e
# MAGIC LEFT JOIN labs l ON e.ENCOUNTER_BK = l.ENCOUNTER_BK;


# MARKDOWN ********************

# ## 5 · Serving (graphql) — the three API objects
# `current_ward_census` (ward board) · `hmo_encounter_summary` (external HMO) · `abnormal_lab_alerts` (machine-to-machine watermark feed)

# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- SERVING (graphql) — current_ward_census   [Scenario 1: live ward board]
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- THE OBJECT THE WARD-BOARD API SERVES. One flat row per currently-admitted
# MAGIC -- patient, with bed location, attending provider, live length of stay, and
# MAGIC -- the "needs extra attention" signals (recurring patient / abnormal labs).
# MAGIC -- Flat + one-row-per-patient = trivial to expose over GraphQL.
# MAGIC --
# MAGIC -- Notes:
# MAGIC --   * "Currently admitted" = encounter not discharged and not cancelled,
# MAGIC --     joined to its ACTIVE bed assignment.
# MAGIC --   * Uses current_date() for live LOS, so this view full-refreshes each run.
# MAGIC --     That's fine: it's tiny (only admitted patients). Schedule it tightly
# MAGIC --     (e.g., every few minutes) for a fresh board.
# MAGIC --   * DQ drops any census row missing a ward/bed so the board never shows a
# MAGIC --     patient with no location.
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW graphql.current_ward_census
# MAGIC (
# MAGIC     CONSTRAINT has_ward CHECK (WARD_NO IS NOT NULL) ON MISMATCH DROP,
# MAGIC     CONSTRAINT has_bed  CHECK (BED_NO  IS NOT NULL) ON MISMATCH DROP
# MAGIC )
# MAGIC COMMENT "Live ward board: one row per currently-admitted patient with location"
# MAGIC AS
# MAGIC WITH admitted AS (
# MAGIC     SELECT *
# MAGIC     FROM silver.encounter_silver
# MAGIC     WHERE DISCHARGE_TS IS NULL
# MAGIC       AND STATUS <> 'Cancelled'
# MAGIC ),
# MAGIC active_bed AS (
# MAGIC     SELECT ENCOUNTER_BK, WARD_NO, BED_NO, ASSIGNED_TS
# MAGIC     FROM silver.bed_assignment_silver
# MAGIC     WHERE IS_ACTIVE
# MAGIC ),
# MAGIC admission_counts AS (                       -- recurring-patient signal
# MAGIC     SELECT PATIENT_BK, count(*) AS ADMISSION_COUNT
# MAGIC     FROM silver.encounter_silver
# MAGIC     GROUP BY PATIENT_BK
# MAGIC ),
# MAGIC abnormal_labs AS (                          -- extra-care signal (this stay)
# MAGIC     SELECT ENCOUNTER_BK, count(*) AS NUM_ABNORMAL_LABS
# MAGIC     FROM silver.lab_order_silver
# MAGIC     WHERE RESULT_FLAG = 'Abnormal'
# MAGIC     GROUP BY ENCOUNTER_BK
# MAGIC )
# MAGIC SELECT
# MAGIC     e.ENCOUNTER_BK,
# MAGIC     e.PATIENT_BK,
# MAGIC     xxhash64(e.PATIENT_BK)                               AS PATIENT_SK,
# MAGIC     p.FULL_NAME,
# MAGIC     p.GENDER,
# MAGIC     p.AGE,
# MAGIC     b.WARD_NO,
# MAGIC     b.BED_NO,
# MAGIC     d.DEPARTMENT_NAME,
# MAGIC     pr.PROVIDER_NAME                                     AS ATTENDING_PROVIDER,
# MAGIC     dx.DIAGNOSIS_DESC,
# MAGIC     dx.DIAGNOSIS_CATEGORY,
# MAGIC     e.ENCOUNTER_TYPE,
# MAGIC     e.ADMIT_TS,
# MAGIC     e.ADMIT_DATE,
# MAGIC     datediff(current_date(), e.ADMIT_DATE)              AS CURRENT_LOS_DAYS,
# MAGIC     coalesce(ac.ADMISSION_COUNT, 1)                     AS ADMISSION_COUNT,
# MAGIC     (coalesce(ac.ADMISSION_COUNT, 1) >= 3)              AS IS_RECURRING,
# MAGIC     coalesce(al.NUM_ABNORMAL_LABS, 0)                   AS NUM_ABNORMAL_LABS,
# MAGIC     -- single flag the app can sort/filter the board by
# MAGIC     (coalesce(ac.ADMISSION_COUNT, 1) >= 3
# MAGIC         OR coalesce(al.NUM_ABNORMAL_LABS, 0) > 0)       AS NEEDS_ATTENTION
# MAGIC FROM admitted e
# MAGIC JOIN active_bed b              ON e.ENCOUNTER_BK = b.ENCOUNTER_BK
# MAGIC LEFT JOIN silver.patient_silver    p  ON e.PATIENT_BK    = p.PATIENT_BK
# MAGIC LEFT JOIN silver.provider_silver   pr ON e.PROVIDER_BK   = pr.PROVIDER_BK
# MAGIC LEFT JOIN silver.department_silver d  ON e.DEPARTMENT_BK = d.DEPARTMENT_BK
# MAGIC LEFT JOIN silver.diagnosis_silver  dx ON e.DIAGNOSIS_BK  = dx.DIAGNOSIS_BK
# MAGIC LEFT JOIN admission_counts ac ON e.PATIENT_BK  = ac.PATIENT_BK
# MAGIC LEFT JOIN abnormal_labs    al ON e.ENCOUNTER_BK = al.ENCOUNTER_BK;

# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- SERVING (graphql) — hmo_encounter_summary   [Scenario 2: external HMO feed]
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- The serving object for the EXTERNAL INSURER (HMO) API. One row per encounter
# MAGIC -- for HMO-covered patients only, billing-focused, carrying HMO_CODE as the
# MAGIC -- scoping key. Lands in the `graphql` serving schema — the API never touches
# MAGIC -- gold/silver directly.
# MAGIC --
# MAGIC -- SCOPING MODEL (the security story):
# MAGIC --   * This MLV holds ALL insurers' rows. It does NOT self-filter to one HMO.
# MAGIC --   * Per-tenant isolation happens at QUERY time: the API injects
# MAGIC --       WHERE HMO_CODE = <the caller's HMO, from the service-principal claim>
# MAGIC --     either via Fabric row-level security or a resolver-side predicate.
# MAGIC --     (That binding is the auth slice you own.)
# MAGIC --   * PARTITIONED BY (HMO_CODE) => that filter is a partition prune: an insurer
# MAGIC --     physically reads only its own partition. Pushdown + isolation in one.
# MAGIC --
# MAGIC -- PII note: deliberately NO patient name here. An external party gets member
# MAGIC -- id + policy + billing, not clinical identity. If a scope needs the name,
# MAGIC -- expose it as a separate GraphQL field gated to authorized callers.
# MAGIC --
# MAGIC -- Refresh: join + filter, no current_date() -> INCREMENTAL-eligible (contrast
# MAGIC -- with current_ward_census, which is full-refresh by design).
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW graphql.hmo_encounter_summary
# MAGIC (
# MAGIC     CONSTRAINT has_hmo_code CHECK (HMO_CODE IS NOT NULL) ON MISMATCH DROP
# MAGIC )
# MAGIC PARTITIONED BY (HMO_CODE)
# MAGIC COMMENT "Per-insurer encounter & billing feed; row-scoped by HMO_CODE"
# MAGIC AS
# MAGIC SELECT
# MAGIC     c.HMO_CODE,                 -- scoping key (partition)
# MAGIC     c.POLICY_NO,
# MAGIC     e.ENCOUNTER_BK,
# MAGIC     e.PATIENT_BK,               -- member id, not name
# MAGIC     e.ADMIT_DATE,
# MAGIC     e.STATUS,
# MAGIC     e.ENCOUNTER_TYPE,
# MAGIC     dx.DIAGNOSIS_DESC,
# MAGIC     dx.DIAGNOSIS_CATEGORY,
# MAGIC     e.BILLED_AMOUNT,
# MAGIC     e.AMOUNT_PAID,
# MAGIC     e.OUTSTANDING_AMOUNT
# MAGIC FROM silver.encounter_silver e
# MAGIC JOIN silver.patient_coverage_silver c
# MAGIC       ON e.PATIENT_BK = c.PATIENT_BK
# MAGIC      AND c.IS_HMO                                  -- HMO members only
# MAGIC LEFT JOIN silver.diagnosis_silver dx
# MAGIC       ON e.DIAGNOSIS_BK = dx.DIAGNOSIS_BK
# MAGIC WHERE e.STATUS <> 'Cancelled';

# CELL ********************

# MAGIC %%sql
# MAGIC -- ===========================================================================
# MAGIC -- SERVING (graphql) — abnormal_lab_alerts   [Scenario 3: M2M watermark feed]
# MAGIC -- ---------------------------------------------------------------------------
# MAGIC -- The serving object for the ABNORMAL-LAB NOTIFICATION feed. No human, no UI:
# MAGIC -- a backend service polls "abnormal results since <timestamp>", routes each to
# MAGIC -- the attending provider, then advances its watermark. Lands in the `graphql`
# MAGIC -- serving schema.
# MAGIC --
# MAGIC -- WATERMARK / since-X MODEL (the new concept this scenario teaches):
# MAGIC --   * Consumers filter WHERE RESULTED_TS > <last-seen ts>. That `since`
# MAGIC --     argument is injected at QUERY time by the resolver/endpoint — the MLV
# MAGIC --     just exposes a precise RESULTED_TS to filter on. (Resolver/auth = yours.)
# MAGIC --
# MAGIC -- WHY APPEND-ONLY (the design lesson):
# MAGIC --   * This feed carries only STABLE facts: a resulted abnormal lab, the
# MAGIC --     attending provider, the department, the diagnosis. A resulted lab never
# MAGIC --     un-results, so RESULTED_TS only ever moves forward and the consumer's
# MAGIC --     watermark is reliable.
# MAGIC --   * It deliberately does NOT join mutable state (current bed, live admit
# MAGIC --     status). Those change after the result lands and would retroactively
# MAGIC --     rewrite old rows, breaking the watermark. If the notifier needs the
# MAGIC --     patient's current location, it calls graphql.current_ward_census.
# MAGIC --   * Append-only + no current_date() => cleanly INCREMENTAL-refresh eligible.
# MAGIC --
# MAGIC -- DQ: only resulted labs are alertable (RESULTED_TS NOT NULL).
# MAGIC -- ===========================================================================
# MAGIC 
# MAGIC CREATE OR REPLACE MATERIALIZED LAKE VIEW graphql.abnormal_lab_alerts
# MAGIC (
# MAGIC     CONSTRAINT is_resulted CHECK (RESULTED_TS IS NOT NULL) ON MISMATCH DROP
# MAGIC )
# MAGIC COMMENT "Append-only abnormal-lab feed; consumers watermark on RESULTED_TS"
# MAGIC AS
# MAGIC SELECT
# MAGIC     l.LAB_ORDER_BK                  AS ALERT_BK,
# MAGIC     l.RESULTED_TS,                  -- the watermark key
# MAGIC     l.LAB_ORDER_BK,
# MAGIC     l.ENCOUNTER_BK,
# MAGIC     e.PATIENT_BK,                   -- member id
# MAGIC     l.TEST_NAME,
# MAGIC     l.RESULT_FLAG,
# MAGIC     e.PROVIDER_BK,                  -- who to route the alert to
# MAGIC     pr.PROVIDER_NAME                AS ATTENDING_PROVIDER,
# MAGIC     d.DEPARTMENT_NAME,
# MAGIC     dx.DIAGNOSIS_DESC,
# MAGIC     e.ADMIT_DATE
# MAGIC FROM silver.lab_order_silver l
# MAGIC JOIN silver.encounter_silver e        ON l.ENCOUNTER_BK  = e.ENCOUNTER_BK
# MAGIC LEFT JOIN silver.provider_silver pr   ON e.PROVIDER_BK   = pr.PROVIDER_BK
# MAGIC LEFT JOIN silver.department_silver d  ON e.DEPARTMENT_BK = d.DEPARTMENT_BK
# MAGIC LEFT JOIN silver.diagnosis_silver dx  ON e.DIAGNOSIS_BK  = dx.DIAGNOSIS_BK
# MAGIC WHERE l.RESULT_FLAG = 'Abnormal'
# MAGIC   AND e.STATUS <> 'Cancelled';
