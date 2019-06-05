#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
WORKS: get shared item
curl https://api.box.com/2.0/shared_items -H "Authorization: Bearer oxk1aa9Ixk4xPnlGx9GcGxKmNAcgJvPl" -H "BoxApi: shared_link=https://anl.box.com/s/acb5au7r5sy6c2mfvkbxcqtv84xtuw4h"
Return:
{"type":"folder","id":"51121818970","sequence_id":"0","etag":"0","name":"cset_retrievals","created_at":"2018-07-10T21:35:31-07:00","modified_at":"2018-08-08T13:45:17-07:00","description":"","size":5704265230,"path_collection":{"total_count":0,"entries":[]},"created_by":{"type":"user","id":"226117525","name":"Ghate, Virendra P.","login":"vghate@anl.gov"},"modified_by":{"type":"user","id":"226117525","name":"Ghate, Virendra P.","login":"vghate@anl.gov"},"trashed_at":null,"purged_at":null,"content_created_at":"2018-07-10T21:35:31-07:00","content_modified_at":"2018-08-08T13:45:17-07:00","owned_by":{"type":"user","id":"226117525","name":"Ghate, Virendra P.","login":"vghate@anl.gov"},"shared_link":null,"folder_upload_email":null,"parent":null,"item_status":"active"}

works: get folder info
curl https://api.box.com/2.0/folders/51121818970 -H "Authorization: Bearer 7XyaPQBKzsmeZ14rwx5GngXdYgY9V35q" -H "BoxApi: shared_link=https://anl.box.com/s/acb5au7r5sy6c2mfvkbxcqtv84xtuw4h"

works: download a file
curl -L https://api.box.com/2.0/files/310392720350/content -H "Authorization: Bearer 7XyaPQBKzsmeZ14rwx5GngXdYgY9V35q" -H "BoxApi: shared_link=https://anl.box.com/s/acb5au7r5sy6c2mfvkbxcqtv84xtuw4h" -o RF02_20150707.cdf

WORKS: list my base folder
curl https://api.box.com/2.0/folders/0 -H "Authorization: Bearer oxk1aa9Ixk4xPnlGx9GcGxKmNAcgJvPl"

WORKS: list test folder
curl https://api.box.com/2.0/folders/52665656274 -H "Authorization: Bearer oxk1aa9Ixk4xPnlGx9GcGxKmNAcgJvPl""""

from subprocess import check_output
import os

null = 'null'
d =  {"type":"folder","id":"51121818970","sequence_id":"0","etag":"0","name":"cset_retrievals","created_at":"2018-07-10T21:35:31-07:00","modified_at":"2018-08-08T13:45:17-07:00","description":"","size":5704265230,"path_collection":{"total_count":0,"entries":[]},"created_by":{"type":"user","id":"226117525","name":"Ghate, Virendra P.","login":"vghate@anl.gov"},"modified_by":{"type":"user","id":"226117525","name":"Ghate, Virendra P.","login":"vghate@anl.gov"},"trashed_at":null,"purged_at":null,"content_created_at":"2018-07-10T21:35:31-07:00","content_modified_at":"2018-08-08T13:45:17-07:00","owned_by":{"type":"user","id":"226117525","name":"Ghate, Virendra P.","login":"vghate@anl.gov"},"shared_link":null,"folder_upload_email":null,"parent":null,"item_status":"active","item_collection":{"total_count":14,"entries":[{"type":"file","id":"310392720350","file_version":{"type":"file_version","id":"327065085134","sha1":"e72e447d68585bd82984193317cad265fb2b6014"},"sequence_id":"0","etag":"0","sha1":"e72e447d68585bd82984193317cad265fb2b6014","name":"RF02_20150707.cdf"},{"type":"file","id":"310412816254","file_version":{"type":"file_version","id":"327088052974","sha1":"95be92ac43a178f5ec79bea6da6e9f45b15b228d"},"sequence_id":"0","etag":"0","sha1":"95be92ac43a178f5ec79bea6da6e9f45b15b228d","name":"RF03_20150709.cdf"},{"type":"file","id":"310413864853","file_version":{"type":"file_version","id":"327089087749","sha1":"be6bb5acd705b1a2038585c498d8641e314ffab6"},"sequence_id":"0","etag":"0","sha1":"be6bb5acd705b1a2038585c498d8641e314ffab6","name":"RF04_20150712.cdf"},{"type":"file","id":"310417512036","file_version":{"type":"file_version","id":"327092657604","sha1":"b25772960f20c106d4faabc492b258487af5d183"},"sequence_id":"0","etag":"0","sha1":"b25772960f20c106d4faabc492b258487af5d183","name":"RF05_20150714.cdf"},{"type":"file","id":"303703286489","file_version":{"type":"file_version","id":"319834831769","sha1":"236d7ea2601103f554420d46b4fe54765cac4a63"},"sequence_id":"0","etag":"0","sha1":"236d7ea2601103f554420d46b4fe54765cac4a63","name":"RF06_20150717.cdf"},{"type":"file","id":"303700240383","file_version":{"type":"file_version","id":"319831498383","sha1":"55565e5feb3da7a2aaca5deee1984a952c529a3a"},"sequence_id":"0","etag":"0","sha1":"55565e5feb3da7a2aaca5deee1984a952c529a3a","name":"RF07_20150719.cdf"},{"type":"file","id":"310414259269","file_version":{"type":"file_version","id":"327089564677","sha1":"d6160f8831186be8d7708bf3064c889170e91e3b"},"sequence_id":"0","etag":"0","sha1":"d6160f8831186be8d7708bf3064c889170e91e3b","name":"RF08_20150722.cdf"},{"type":"file","id":"310422537181","file_version":{"type":"file_version","id":"327098569645","sha1":"b2704e9a135e0067e5015ac6961e4e9292f33956"},"sequence_id":"0","etag":"0","sha1":"b2704e9a135e0067e5015ac6961e4e9292f33956","name":"RF09_20150724.cdf"},{"type":"file","id":"303705891133","file_version":{"type":"file_version","id":"319837176349","sha1":"5bf9cd647b666fc33e5fff572d73046d927a0f9c"},"sequence_id":"0","etag":"0","sha1":"5bf9cd647b666fc33e5fff572d73046d927a0f9c","name":"RF10_20150727.cdf"},{"type":"file","id":"303702614434","file_version":{"type":"file_version","id":"319833932914","sha1":"e8fb13dcd406ac6e10a67966d7686f9d82dac8c9"},"sequence_id":"0","etag":"0","sha1":"e8fb13dcd406ac6e10a67966d7686f9d82dac8c9","name":"RF11_20150729.cdf"},{"type":"file","id":"310417975490","file_version":{"type":"file_version","id":"327093172466","sha1":"7275942cc92b8fcab5b8b6a4c04d7c2997d9f7be"},"sequence_id":"0","etag":"0","sha1":"7275942cc92b8fcab5b8b6a4c04d7c2997d9f7be","name":"RF12_20150801.cdf"},{"type":"file","id":"310418935740","file_version":{"type":"file_version","id":"327094175484","sha1":"04a96d99c601a16dbd8689af87aa8772e4b00432"},"sequence_id":"0","etag":"0","sha1":"04a96d99c601a16dbd8689af87aa8772e4b00432","name":"RF13_20150803.cdf"},{"type":"file","id":"310416424276","file_version":{"type":"file_version","id":"327091705924","sha1":"f5b2fd02b041dbca76ec6c51e16fdbe35e0e524b"},"sequence_id":"0","etag":"0","sha1":"f5b2fd02b041dbca76ec6c51e16fdbe35e0e524b","name":"RF14_20150807.cdf"},{"type":"file","id":"310416776657","file_version":{"type":"file_version","id":"327092124833","sha1":"e61ada2760af5977bebb1dc8711725d292e31d85"},"sequence_id":"0","etag":"0","sha1":"e61ada2760af5977bebb1dc8711725d292e31d85","name":"RF15_20150809.cdf"}],"offset":0,"limit":100,"order":[{"by":"type","direction":"ASC"},{"by":"name","direction":"ASC"}]}}

for v in d["item_collection"]["entries"]:
    
    curl_call = 'curl -L https://api.box.com/2.0/files/{}/content -H "Authorization: Bearer 7XyaPQBKzsmeZ14rwx5GngXdYgY9V35q" -H "BoxApi: shared_link=https://anl.box.com/s/acb5au7r5sy6c2mfvkbxcqtv84xtuw4h" -o {}'.format(v['id'], v['name'])
    print(curl_call)

    check_output(curl_call, shell=True, cwd = '/home/disk/eos4/jkcm/Data/CSET/virendra/box')
