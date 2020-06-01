
import requests

url = 'http://127.0.0.1:5000/'
r = requests.post(url,json={'time_in_hoapital':1,'num_lab_procedures':1 ,'num_medications':1 ,'number_outpatient':1,'number_emergency':1,'number_inpatient':1 ,'number_diagnoses':1 })

print(r.json())
