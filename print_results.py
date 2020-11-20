import json

f = open('sample results.json')
result = json.load(f)
#for i in result['episode_rewards']:
#	print(i)
print(result)	
f.close
