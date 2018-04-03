import sys
import json

main = json.load(open(sys.argv[1]))
add = json.load(open(sys.argv[2]))

for k in add.keys():
    main[k] = add[k]

with open(sys.argv[1], 'w') as f:
    json.dump(main, fp=f)

