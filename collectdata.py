#from BML.data import Dataset
from BML.transform import DatasetTransformation
from BML import utils
#################
# Data collection
folder = "dataset/"
dataset = Dataset(folder)
dataset.load({
"PrimingPeriod": 10*60*60, # 10 hours of priming data
"IpVersion": [4], # only IPv4 routes
"Collectors": ["rrc06"], # rrc06: at Otemachi, Japan
"PeriodsOfInterests":
[{
"name": "GoogleLeak",
"label": "anomaly",
"start_time": utils.getTimestamp(2017, 8, 25, 3, 0, 0),
# August 25, 2017, 3:00 UTC
"end_time": utils.getTimestamp(2017, 8, 25, 4, 0, 0)
# August 25, 2017, 4:00 UTC
}]
})
# run the data collection
utils.runJobs(dataset.getJobs(), folder+"collect_jobs")
#####################
# Data transformation
# features extraction every minute
datTran = DatasetTransformation(folder,
"BML.transform", "Features", 1)
# run the data transformation
utils.runJobs(datTran.getJobs(), folder+"transform_jobs")