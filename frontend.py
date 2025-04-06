import streamlit as st
from gesture_controller import GestureControllerApp
import json
from streamlit_lottie import st_lottie
import time# Assuming the original logic is saved in a separate file
# Title for the Streamlit app
app = GestureControllerApp()

flag=0
st.markdown("""
    <h1 style="text-align: center;font-size:70px;"> Touchless Control
    </h1>
            <p style="text-align:center;font-size:20px;">Control your digital world with just a gesture</p>
            
""", unsafe_allow_html=True)
col1, col2, col3= st.columns([1, 1, 1])  # Adjust the proportions as needed
# Place the button in the center column
with col2:
    st.write(" ")
    st.write(" ")
    start_button=st.button("Start virtual control")
if start_button:
    app.start()
    st.write("Gesture Controller started. Use gestures to interact.Press q on keyboard to exit")
    st.write(" ")
def load_lottie_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
lottie_animation = load_lottie_file("Animation - 1742730378103.json")
st_lottie(lottie_animation,speed=1,reverse=False,loop=True,height=500)
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")

st.markdown("""
    <h1 style="text-align: center;font-size:50px;">About The Project
    </h1>
            <p style="text-align:center;font-size:20px;">A revolutionary gesture-based interface that eliminates the need for physical touch, providing seamless interaction with your digital workspace.</p>
""", unsafe_allow_html=True)
st.title("")
st.markdown("""
    <h1 style="text-align: center;font-size:50px;">Instructions
    </h1>
""", unsafe_allow_html=True)
st.text(" ")
st.text(" ")
# Create three columns for the tiles
col1, col2,col3 = st.columns([2, 1,2])

# Add content to the first tile with a border
with col1:
    st.markdown("""
        <div style="border: 2px solid #8b5cf6; border-radius: 10px; padding: 20px; text-align: center;">
            <h3 style="font-size:69px;line-height: 100px;">1</h3>
            <p>Setup your Camera <br> </p>
        </div>
    """, unsafe_allow_html=True)
    st.text("")
    st.text("")
    c1, c2,c3 = st.columns([1,3,1])
    with c2:
        test_button=st.button("test if camera works properly")
    if test_button:    
    # Placeholder to conditionally show the camera input
        camera_placeholder = st.empty()
    
    # Display the camera input for 3 seconds
        picture = camera_placeholder.camera_input("Testing")

    # Wait for 3 seconds before removing the camera input
        time.sleep(3)
        camera_placeholder.empty() 

# Add content to the third tile with a border
with col3:
    st.markdown("""
        <div style="border: 2px solid #ff5733; border-radius: 10px; padding: 20px; text-align: center;">
            <h3 style="font-size:69px;line-height: 100px;">2</h3>
            <p>learn the gestures</p>
        </div>
    """, unsafe_allow_html=True)
    st.text("")
    st.text("")
    c1, c2,c3 = st.columns([1,3,1])
    with c2:
        tutorial=st.button("scroll down for gesture tutorial")

st.write(" ")
col1, col2, col3 = st.columns([1, 1.8, 1])
# Add content to the second tile with a border
with col2:
    st.markdown("""
        <div style="border: 2px solid #4caf50; border-radius: 10px; padding: 20px; text-align: center;">
            <h3 style="font-size:69px;line-height: 100px;">3</h3>
            <p>start creating</p>
        </div>
    """, unsafe_allow_html=True)
    c1, c2,c3 = st.columns([2,3,2])
    
    with c2:
        st.text("")
        st.text("")

        if st.button("Get Started"):
            app.start()

st.markdown("""
    <h1 style="text-align: center;font-size:70px;"> Tutorial
    </h1>   
""", unsafe_allow_html=True)
c1, c2,c3 = st.columns(3)


with c1:
    st.image("index.png",caption="cursor")

with c2:
    st.image("erase.png",caption="erase/scroll")
with c3:
    st.image("copy.png",caption="copy")
with c1:
    st.image("paste.png",caption="paste")
with c2:
    st.image("pinky.png",caption="whiteboard toggle")
with c3:
    st.image("control toggle.png",caption="on or off the virtual control")

st.title("")
st.markdown("""
    <h1 style="text-align: center;font-size:70px;"> Our Team
    </h1>            
""", unsafe_allow_html=True)
st.write("")
st.write("")
st.write("")
st.write("")

col1, col2,col3,col4 = st.columns(4)
with col1:
    st.image("https://media.licdn.com/dms/image/v2/D5603AQE1U1FRmDqmrQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1700895930496?e=1748476800&v=beta&t=50t4KeY0r__pD7_53eXrqOuXOoU52IQO1MyMqzqSJBQ")
# Add clickable text with a link
    st.markdown("""
        <p style="text-align: center; font-size: 16px;"><a href="https://www.linkedin.com/in/aldrin-binu-801710259/" target="_blank" style="color: #8b5cf6; text-decoration: none; font-weight: bold;">Aldrin Binu</a>
        </p>
    """, unsafe_allow_html=True)

# Add content to the third tile with a border
with col2:
    st.image("https://media.licdn.com/dms/image/v2/D5603AQE1esw-D7QyIw/profile-displayphoto-shrink_800_800/B56ZXDJ.gCHoAc-/0/1742735931441?e=1748476800&v=beta&t=hZEj6IpqxAbI4mV-XSGBoLjGTuple1YI27GyRFqrTKI")
    st.markdown("""
        <p style="text-align: center; font-size: 16px;"><a href="https://www.linkedin.com/in/aneena-m-s-693895266/" target="_blank" style="color: #8b5cf6; text-decoration: none; font-weight: bold;">Aneena Ms</a>
        </p>
    """, unsafe_allow_html=True)
with col3:
   st.image("https://media.licdn.com/dms/image/v2/D5603AQFaAKw1uGMRgg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1732295518063?e=1748476800&v=beta&t=N99MdGBWXJ1NCTek1ZyjUn811jJekb0rYSpGGLoxrGk")
   st.markdown("""
        <p style="text-align: center; font-size: 16px;"><a href="https://www.linkedin.com/in/namitha-anna-koshy-67096727b/" target="_blank" style="color: #8b5cf6; text-decoration: none; font-weight: bold;">Namitha Anna Koshy</a>
        </p>
    """, unsafe_allow_html=True)
# Add content to the third tile with a border
with col4:
    st.image("https://media.licdn.com/dms/image/v2/D5603AQEt0tUXmnwLtg/profile-displayphoto-shrink_800_800/B56ZSZNrYDGUAc-/0/1737737289247?e=1748476800&v=beta&t=kDtvcIP-1P33ahvwXxg102TvLJ968RSg8u6utAklyHQ")
    st.markdown("""
        <p style="text-align: center; font-size: 16px;"><a href="https://www.linkedin.com/in/rithika-krishna-496639250/" target="_blank" style="color: #8b5cf6; text-decoration: none; font-weight: bold;">Rithika Krishna</a>
        </p>
    """, unsafe_allow_html=True)
st.title("")

   