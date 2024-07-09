import httpx
import datetime
import time
import toml
from celery_app import app

def clean_old_studies():
    config = toml.load("config.toml")
    cleanup_settings = config['cleanup_settings']
    base_url = cleanup_settings['base_url']
    timeout_days = cleanup_settings['timeout_days']
    whitelist = cleanup_settings['whitelist']
    sleep_duration = cleanup_settings['sleep_duration']

    response = httpx.get(f'{base_url}/studies')
    study_ids = response.json()

    now = datetime.datetime.now()

    for study_id in study_ids:
        study_url = f"{base_url}/studies/{study_id}"
        time.sleep(sleep_duration)
        study_response = httpx.get(study_url)
        study_data = study_response.json()
        
        last_update_str = study_data.get("LastUpdate")
        study_instance_uid_str = study_data.get("MainDicomTags").get("StudyInstanceUID")
        patient_id_str = study_data.get("PatientMainDicomTags").get("PatientID")
        patient_name_str = study_data.get("PatientMainDicomTags").get("PatientName")
        last_update_time = datetime.datetime.strptime(last_update_str, "%Y%m%dT%H%M%S")
        
        time_diff = now - last_update_time
        
        print(f"\nStudy ID: {study_id}")
        print(f"StudyInstanceUID: {study_instance_uid_str}")
        # print(f"Details: {study_data}")
        print(f"Days since last update: {time_diff.days} days")

        if study_id in whitelist or study_instance_uid_str in whitelist or patient_id_str in whitelist or patient_name_str in whitelist:
            print(f"Study {study_id} / {study_instance_uid_str} is in the whitelist. Skipping deletion.")
            continue

        if time_diff.days > timeout_days:
            time.sleep(sleep_duration)
            delete_response = httpx.delete(study_url)
            if delete_response.status_code == 200:
                print(f"Deleted study {study_id} / {study_instance_uid_str} !!!!!!!")
            else:
                print(f"Failed to delete study {study_id} / {study_instance_uid_str}")
        else:
            print(f"Study {study_id} / {study_instance_uid_str} is within the allowed time range.")
            # print(f"Stopping execution.")
            # break


# clean_old_studies()
@app.task
def run_clean_old_studies():
    clean_old_studies()
    