import streamlit as st
import base64 

def set_bg_hack():
    main_bg = "background.png"
    main_bg_ext = "png"
    
    bg="backg.jpg"
    bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .appview-container {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
         }}
         .block-container {{
             background: url(data:image/{bg_ext};base64,{base64.b64encode(open(bg, "rb").read()).decode()})
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
        
def header1(text):
    html_temp = f"""
    <h1 style = "color:brown; text_align:center; font-weight: bold;"> {text} </h2>
    </div>
    """   
    st.markdown(html_temp, unsafe_allow_html = True)
    
def header2(text):
    html_temp = f"""
    <h3 style = "color:brown; text_align:center; font-weight: bold;"> {text} </h2>
    </div>
    """   
    st.markdown(html_temp, unsafe_allow_html = True)

def header3(text):
    html_temp = f"""
    <h5 style = "color:brown; text_align:center; font-weight: bold;"> {text} </h2>
    </div>
    """   
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
