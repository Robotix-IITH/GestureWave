import socket
import struct
import unicodedata
import csv
import json
import os
import requests
import subprocess
try:
    from types import SimpleNamespace as Namespace
except ImportError:
    from argparse import Namespace



def run_bash_script(data):
    
    
    
    try:
        # Construct the command to run the Bash script with the data
        command = f'bash testing_file.sh "{data}"'
        # Execute the Bash script with the data
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Bash script executed successfully")
            print("Output:")
            print(result.stdout)
        else:
            print("Bash script failed to execute")
            print("Error:")
            print(result.stderr)
    except Exception as e:
        print(f"Error: {e}")

    return result.stdout
# data settings
data_size = 138 # sending 16 bytes = 128 bits (binary touch states, for example)

# server settings
#server_name = "192.168.0.100"
server_name = "172.20.10.3"
server_port = 5001
server_address = (server_name, server_port)
x_val = 0
# start up server
print ('Setting up server on:', server_address)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(server_address)
server_socket.listen(1)

# wait for connection
print ('Waiting for a client connection...')
connection, client_address = server_socket.accept()
print ('Connected to:', client_address)




count = 0

session = 0
gesture_recgo = ''
while True:
    final_data = []
    while True:
        # read length of data (4 bytes)
        try:
            count += 1
            length_bytes = connection.recv(4)
            if not length_bytes:
                break
            length = struct.unpack('>I', length_bytes)[0]
            # read data itself
            data = connection.recv(length)
            if not data:
                break
        
            if data.decode('utf-8') == "done!" :
                print("reset")
                break
            
            # if data.decode('utf-8') == "gesture_name":
            #     gestureName = data.decode('utf-8')
            #     os.makdirs(str(gestureName))
            # do something with data
            print("////////////////////////////////////////////////////////////////////////////////////")
            print(count)
            print("Received data from client:", data.decode('utf-8'))
            result = unicodedata.normalize('NFKD', data.decode('utf-8')).encode('ascii', 'ignore')
            result_obj =  json.loads(result, object_hook=lambda d: Namespace(**d))
            
            a_x_str = unicodedata.normalize('NFKD', result_obj.x_val).encode('ascii', 'ignore')
            a_y_str = unicodedata.normalize('NFKD', result_obj.y_val).encode('ascii', 'ignore')
            a_z_str = unicodedata.normalize('NFKD', result_obj.z_val).encode('ascii', 'ignore')
            
            g_x_str = unicodedata.normalize('NFKD', result_obj.xG_val).encode('ascii', 'ignore')
            g_y_str = unicodedata.normalize('NFKD', result_obj.yG_val).encode('ascii', 'ignore')
            g_z_str = unicodedata.normalize('NFKD', result_obj.zG_val).encode('ascii', 'ignore')
            
            x_L_str = unicodedata.normalize('NFKD', result_obj.xL_val).encode('ascii', 'ignore')
            y_L_str = unicodedata.normalize('NFKD', result_obj.yL_val).encode('ascii', 'ignore')
            z_L_str = unicodedata.normalize('NFKD', result_obj.zL_val).encode('ascii', 'ignore')
            

            a_norm_str = unicodedata.normalize('NFKD', result_obj.a_Mag).encode('ascii', 'ignore')
            g_norm_str = unicodedata.normalize('NFKD', result_obj.g_Mag).encode('ascii', 'ignore')
            l_norm_str = unicodedata.normalize('NFKD', result_obj.l_Mag).encode('ascii', 'ignore')
            final_data.append([float(a_x_str), float(a_y_str),float(a_z_str) ,float(g_x_str), float(g_y_str),float(x_L_str),float(y_L_str),float(z_L_str), float(g_z_str), float(a_norm_str), float(g_norm_str), float(l_norm_str)])
            print(final_data) 
            gesture = unicodedata.normalize('NFKD', result_obj.gesture_name).encode('ascii', 'ignore').decode('utf-8')
            gesture_final = "./testing_data/"+gesture
            if not os.path.exists(gesture_final):
                os.makedirs(gesture_final)
                print(f"Directory '{gesture_final}' created successfully!")
            else: 
                print("not working")
            
        except:
            continue
    
    print("the final is ",final_data)

    if len(final_data) == 0:
        print("client disconneted from server" , final_data)
        session = 0
        break
    print("Session done")
    session+= 1
    #loc = "./testing_data/"+str(gesture)+"/"+str(gesture)+str(session)+".csv"
    with open("./testing_data/"+str(gesture)+"/"+str(gesture)+str(session)+".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerows(final_data)
    print("file saved")
    if gesture == "testing":
        loc = "./testing_data/"+str(gesture)+"/"+str(gesture)+str(session)+".csv"
        gesture_recgo= run_bash_script(loc)

  
    myobj = {'gesture': gesture_recgo}
    url = 'http://172.20.10.2:3000/api/gestures'  # IP address of the other machine running the Flask app
    headers = {'Content-type': 'application/json'}
    #response = request.post(url, json={'test': test}, headers=headers)
    requests.post(url, json = myobj, headers=headers)


server_socket.close()
