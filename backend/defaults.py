from backend.schemas import Venue


def default_venue():
    names = ["North concourse","North hall","East concourse","West hall","Central atrium","East hall","West approach","South concourse","South approach"]
    zones=[]
    for i,name in enumerate(names):
        row,col=divmod(i,3)
        x,y=65+col*290,100+row*255
        zones.append({"id":f"Z{i+1:02}","name":name,"polygon":[[x,y],[x+265,y],[x+265,y+225],[x,y+225]],"image_polygon":[[col*1000/3,row*1000/3],[(col+1)*1000/3,row*1000/3],[(col+1)*1000/3,(row+1)*1000/3],[col*1000/3,(row+1)*1000/3]],"capacity":100,"area_m2":None})
    portals=[]
    for i,z in enumerate(zones):
        row,col=divmod(i,3)
        for j in ([i+1] if col<2 else [])+([i+3] if row<2 else []):
            a,b=z["polygon"][0],zones[j]["polygon"][0]
            portals.append({"id":f"P{i+1}-{j+1}","name":f"Passage {i+1}–{j+1}","source":z["id"],"target":zones[j]["id"],"kind":"passage","position":[(a[0]+b[0])/2+132,(a[1]+b[1])/2+112],"capacity":3,"bidirectional":True,"travel_seconds":8})
    portals += [
        {"id":"E-N","name":"North exit","source":"Z02","target":"outside","kind":"exit","position":[487,70],"capacity":1.8},
        {"id":"E-E","name":"East exit","source":"Z06","target":"outside","kind":"exit","position":[940,467],"capacity":1.4},
        {"id":"E-S","name":"South exit","source":"Z08","target":"outside","kind":"exit","position":[487,865],"capacity":2.2},
        {"id":"I-W","name":"West entrance","source":"outside","target":"Z04","kind":"entrance","position":[30,467],"capacity":1.5},
    ]
    return Venue(id="atrium-study",name="Atrium · scenario study",description="Illustrative nine-zone layout. Gate geometry and capacities are assumptions, not recovered from video.",geometry_kind="schematic",zones=zones,portals=portals).model_dump()

