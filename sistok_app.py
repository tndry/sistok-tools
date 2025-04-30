import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from datetime import datetime
import matplotlib.pyplot as plt
import os
import gdown
from sklearn.linear_model import LinearRegression




# Konfigurasi layout Streamlit
st.set_page_config(
    page_title="Sistok App",
    page_icon="🐟",
    layout="wide"
)
# # Inisialisasi OpenAI Client
# client = OpenAI(
#     api_key=st.secrets.DEEPSEEK,
#     base_url="https://api.deepseek.com"
           
#                 )

# Data ASLIII
# # ID file Googel Drive
# file_id = '1eACQIHOn3oS96V8rHzN6VlMuKtNX5raz'
# drive_url = f'https://drive.google.com/uc?id={file_id}'


# Data DEMOO
# ID file Googel Drive
file_id = '1wXxn-GJtVfEaZJTH-IPe4svur9uLbZIs' 
drive_url = f'https://drive.google.com/uc?id={file_id}'



# Fungsi untuk memuat data dari database atau file CSV
@st.cache_data
def load_data():
    try:
        # Download file CSV
        # ASLI
        file_path = 'data_bersih.csv'  
        #DEMO
        file_path = 'data/data_bersih_demo.csv'
        gdown.download(drive_url, file_path, quiet=False)
        # Baca file CCSV
        df = pd.read_csv(drive_url,  low_memory=False) #kalo mau gunain demo pake yg file_path
        # Konversi tanggal ke tipe datetime
        df['tanggal_berangkat'] = pd.to_datetime(df['tanggal_berangkat'], errors='coerce')
        df['tanggal_kedatangan'] = pd.to_datetime(df['tanggal_kedatangan'], errors='coerce')
        df['tahun'] = df['tanggal_kedatangan'].dt.year

        
        return df

    except FileNotFoundError:
        st.error("File tidak ditemukan. Pastikan file 'data_bersih.csv' ada di folder './data/'." )
        return pd.DataFrame()

# Fungsi filter data
def filter_data(df, pelabuhan_kedatangan_id, nama_ikan_id, start_year, end_year, time_frame):

    # Filter data berdasarkan pelabuhan kedatangan
    if pelabuhan_kedatangan_id:
        df = df[df['pelabuhan_kedatangan_id'] == pelabuhan_kedatangan_id]

    # Filter data berdasarkan nama ikan
    if nama_ikan_id:
        df = df[df['nama_ikan_id'].isin(nama_ikan_id)]

    # Filter berdasarkan tahun
    if start_year:
        df = df[df['tahun'] >= start_year]
    if end_year:
        df = df[df['tahun'] <= end_year]

    # Filter berdasarkan time frame
    if time_frame == 'Daily':
        df['time_period'] = df['tanggal_kedatangan'].dt.date
    elif time_frame == 'Weekly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('W').astype(str)
    elif time_frame == 'Monthly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('M').astype(str)
    elif time_frame == 'Yearly':
        df['time_period'] = df['tanggal_kedatangan'].dt.to_period('Y').astype(str)


    return df

# Function to get OpenAI chat response

def analyze_fishing_data(query, filtered_data):
    """
    Fungsi untuk menganalisis data perikanan berdasarkan query pengguna
    """
    query = query.lower()
    response = ""
    
    try:
        # Analisis total tangkapan
        if 'total tangkapan' in query or 'berapa tangkapan' in query:
            # Filter berdasarkan jenis ikan jika disebutkan
            for fish in filtered_data['nama_ikan_id'].unique():
                if fish.lower() in query:
                    specific_data = filtered_data[filtered_data['nama_ikan_id'].str.lower() == fish.lower()]
                    
                    # Filter tahun jika disebutkan
                    for year in filtered_data['tahun'].unique():
                        if str(year) in query:
                            year_data = specific_data[specific_data['tahun'] == year]
                            total = year_data['berat'].sum()
                            return f"Total tangkapan {fish} pada tahun {year} adalah {total:,.2f} Kg"
                    
                    # Jika tahun tidak disebutkan, tampilkan semua tahun
                    yearly_data = specific_data.groupby('tahun')['berat'].sum()
                    response = f"Total tangkapan {fish} per tahun:\n"
                    for year, total in yearly_data.items():
                        response += f"Tahun {year}: {total:,.2f} Kg\n"
                    return response

        # Analisis alat tangkap
        elif 'alat tangkap' in query or 'jenis alat' in query:
            alat_tangkap = filtered_data.groupby('jenis_api')['berat'].sum().sort_values(ascending=False)
            response = "Alat tangkap yang digunakan (berdasarkan total tangkapan):\n"
            for alat, total in alat_tangkap.items():
                response += f"{alat}: {total:,.2f} Kg\n"
            return response

        # Analisis tren tahunan
        elif 'tren' in query or 'perkembangan' in query:
            yearly_trend = filtered_data.groupby('tahun')['berat'].sum()
            max_year = yearly_trend.idxmax()
            min_year = yearly_trend.idxmin()
            
            response = "Analisis tren tangkapan:\n"
            response += f"Tahun dengan tangkapan tertinggi: {max_year} ({yearly_trend[max_year]:,.2f} Kg)\n"
            response += f"Tahun dengan tangkapan terendah: {min_year} ({yearly_trend[min_year]:,.2f} Kg)\n"
            
            # Hitung pertumbuhan year-over-year
            yoy_growth = yearly_trend.pct_change() * 100
            response += "\nPertumbuhan year-over-year:\n"
            for year, growth in yoy_growth.items():
                if not pd.isna(growth):
                    response += f"{year}: {growth:,.1f}%\n"
            return response

        # Analisis nilai produksi
        elif 'nilai produksi' in query or 'nilai ekonomi' in query:
            if 'tahun' in query:
                for year in filtered_data['tahun'].unique():
                    if str(year) in query:
                        year_data = filtered_data[filtered_data['tahun'] == year]
                        total_value = year_data['nilai_produksi'].sum()
                        return f"Total nilai produksi tahun {year}: Rp {total_value:,.2f}"
            
            total_value = filtered_data['nilai_produksi'].sum()
            avg_value = filtered_data.groupby('tahun')['nilai_produksi'].mean()
            response = f"Total nilai produksi: Rp {total_value:,.2f}\n"
            response += "Rata-rata nilai produksi per tahun:\n"
            for year, value in avg_value.items():
                response += f"Tahun {year}: Rp {value:,.2f}\n"
            return response

        # Analisis jenis ikan
        elif 'jenis ikan' in query or 'ikan apa' in query:
            top_fish = filtered_data.groupby('nama_ikan_id')['berat'].sum().sort_values(ascending=False).head(5)
            response = "5 jenis ikan dengan tangkapan terbanyak:\n"
            for fish, total in top_fish.items():
                response += f"{fish}: {total:,.2f} Kg\n"
            return response

        # Default response
        else:
            return """Saya dapat membantu Anda menganalisis data perikanan. Anda dapat bertanya tentang:
                1 . Total tangkapan (per jenis ikan/tahun)
                2. Alat tangkap yang digunakan
                3. Tren tangkapan tahunan
                4. Nilai produksi
                5. Jenis ikan dominan

                Contoh: 'Berapa total tangkapan cumi tahun 2022?' atau 'Apa saja alat tangkap yang digunakan?'"""

    except Exception as e:
        return f"Maaf, terjadi kesalahan dalam menganalisis data: {str(e)}"

# Ganti fungsi get_openai_response dengan fungsi ini
def get_openai_response(query, filtered_data):
    return analyze_fishing_data(query, filtered_data)

# st.write('Kolom yang ada:', data.columns)


# CSS
st.markdown(
    """
    <style>
    .metric-box{
      border: 1px solid #ccc;
      padding: 10px;
      border-radius: 5px;
    
      margin: 5px;
      text-align: center;
    }
    </style>
""", unsafe_allow_html=True
)

# Initialize chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown("<h1 style='text-align: center; '>SISTOK</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; '>Fish Stock Analysis Tools</h2>", unsafe_allow_html=True)



# Menu
menu = option_menu(None, ['Dashboard', 'Analysis', 'About'],
    icons= ['house', 'graph-up', 'book'],
    menu_icon='cast', default_index=0, orientation='horizontal')

# # Sidebar untuk navigasi
# menu = st.sidebar.radio('Navigasi', ['Dashboard', 'Analysis', 'About'])

# Memuat data
data = load_data()

if menu == 'Dashboard':
    # Custom CSS for more elegant styling
    st.markdown("""
    <style>
        /* Main page styling */
        .main-header {
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            color: #1E88E5;
            padding-bottom: 20px;
            text-align: center;
            border-bottom: 2px solid #f0f2f6;
            margin-bottom: 30px;
        }
        
        /* Metrics styling */
        .metric-container {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }
        
        .metric-box {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            transition: transform 0.3s ease;
            color: #333;
        }
        
        .metric-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
        }
        
        .metric-icon {
            font-size: 28px;
            margin-bottom: 10px;
            color: #1E88E5;
        }
        
        .metric-title {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.4em;
            font-weight: bold;
            color: #1E88E5;
        }
        
        /* Chart container styling */
        .chart-container {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Filter panel styling */
        .sidebar .stSelectbox, .sidebar .stMultiSelect, .sidebar .stNumberInput {
            margin-bottom: 20px;
        }
        
        /* Chat styling */
        .chat-container {
            background-color: #f9f9f9;
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .user-message {
            background-color: #DCF8C6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        .assistant-message {
            background-color: #ECECEC;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        
        /* Data preview styling */
        .preview-header {
            font-weight: 600;
            color: #1E88E5;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        /* Warning and success message styling */
        .custom-warning {
            background-color: #FFF3CD;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #FFD166;
            margin-bottom: 20px;
        }
        
        .custom-success {
            background-color: #D4EDDA;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            border-left: 5px solid #8BBF9F;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    

    # Sidebar Filtering with improved styling
    st.sidebar.markdown("<h3 style='text-align: center; color: #1E88E5;'>Filter Data</h3>", unsafe_allow_html=True)
    
    # Add a separator line
    st.sidebar.markdown("<hr style='margin: 15px 0px; border: none; height: 1px; background-color: #f0f2f6;'>", unsafe_allow_html=True)
    
    pelabuhan = st.sidebar.selectbox("Pilih Pelabuhan", options=[None] + list(data['pelabuhan_kedatangan_id'].unique()))

    jenis_ikan = st.sidebar.multiselect("Pilih Jenis Ikan", options=list(data['nama_ikan_id'].unique()), default=[])

    start_year = st.sidebar.number_input('Start Year', min_value=int(data['tahun'].min()), max_value=int(data['tahun'].max()), value=int(data['tahun'].min()), step=1)
    
    end_year = st.sidebar.number_input('End Year', min_value=start_year, max_value=int(data['tahun'].max()), value=int(data['tahun'].max()), step=1)

    time_frame = st.sidebar.selectbox('Time Frame', ['Daily', 'Weekly', 'Monthly', 'Yearly'])

    # Filter data
    filtered_data = filter_data(data, pelabuhan, jenis_ikan, start_year, end_year, time_frame)

    # Chatbot with improved styling
    st.sidebar.markdown("<hr style='margin: 25px 0px; border: none; height: 1px; background-color: #f0f2f6;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 style='text-align: center; color: #1E88E5;'>Ask AI</h3>", unsafe_allow_html=True)

    # Chat input with better styling
    user_input = st.sidebar.text_input('Ask about the Data:', key='chat_input', placeholder="Type your question here...")

    # Send Button with improved styling
    if st.sidebar.button('Send', key='send_button', use_container_width=True):
        if user_input:
            # Add user message to history
            st.session_state.chat_history.append({'role': 'user', 'content': user_input})

            # Get bot response
            with st.spinner('Thinking...'):
                bot_response = get_openai_response(user_input, filtered_data)
            # Add bot response to history
            st.session_state.chat_history.append({'role': 'assistant', 'content': bot_response})

    # Display chat history with improved styling
    st.sidebar.markdown("<h4 style='color: #666; margin-top: 20px;'>Riwayat Chat</h4>", unsafe_allow_html=True)
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.sidebar.markdown(f"""
            <div class='user-message'>
                <b>Anda:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown(f"""
            <div class='assistant-message'>
                <b>Assistant:</b><br>{message['content']}
            </div>
            """, unsafe_allow_html=True)

    # Clear chat history button with better styling
    if st.sidebar.button("Hapus Riwayat Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Data completeness check with improved styling
    if 2024 in range(start_year, end_year+1):
        if not filtered_data.empty:
            data_tahun_2024 = filtered_data[filtered_data['tahun'] == 2024]
            if data_tahun_2024.empty or data_tahun_2024['berat'].sum() == 0:
                st.markdown("""
                <div class='custom-warning'>
                    ⚠️ Data tahun 2024 belum lengkap. Mohon diperhatikan!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='custom-success'>
                    ✅ Data tahun 2024 sudah lengkap.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='custom-warning'>
                ⚠️ Data tidak tersedia. Silahkan periksa kembali filter Anda.
            </div>
            """, unsafe_allow_html=True)

    # Rename column
    columns_to_rename = {
        'nilai_produksi': 'Nilai Produksi',
        'jumlah_hari': 'Jumlah Hari',
        'pelabuhan_kedatangan_id': 'Pelabuhan Kedatangan',
        'pelabuhan_keberangkatan_id': 'Pelabuhan Keberangkatan',
        'kelas_pelabuhan': 'Port Class',
        'provinsi': 'Provinsi',
        'tanggal_berangkat': 'Tanggal Berangkat',
        'tanggal_kedatangan': 'Tanggal Kedatangan',
    }
    # Rename columns that exist in dataframe
    filtered_data = filtered_data.rename(columns={k: v for k, v in columns_to_rename.items() if k in filtered_data.columns})

    # Compute top analytics
    if not filtered_data.empty:
        total_tangkapan = float(pd.Series(filtered_data['berat']).sum())
        total_nilai_produksi = float(pd.Series(filtered_data['Nilai Produksi']).sum())
        total_hari = filtered_data['Jumlah Hari'].sum()
        total_ikan = filtered_data['nama_ikan_id'].nunique()
    else:
        total_tangkapan = total_nilai_produksi = total_hari = total_ikan = 0

    # Display top analytics with improved styling
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    
    total1, total2, total3, total4 = st.columns(4, gap='small')
    
    with total1:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>🐟</div>
            <div class='metric-title'>Total Tangkapan</div>
            <div class='metric-value'>{total_tangkapan:,.0f} Kg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with total2:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>💵</div>
            <div class='metric-title'>Nilai Produksi</div>
            <div class='metric-value'>{total_nilai_produksi:,.0f} IDR</div>
        </div>
        """, unsafe_allow_html=True)
    
    with total3:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>📆</div>
            <div class='metric-title'>Total Hari</div>
            <div class='metric-value'>{total_hari}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with total4:
        st.markdown(f"""
        <div class='metric-box'>
            <div class='metric-icon'>🎣</div>
            <div class='metric-title'>Jenis Ikan</div>
            <div class='metric-value'>{total_ikan}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Improved data preview section
    with st.expander('PREVIEW DATASET', expanded=False):
        st.markdown("<p class='preview-header'>Select columns to display:</p>", unsafe_allow_html=True)
        showData = st.multiselect('Filter:', filtered_data.columns, default=filtered_data.columns)
        st.dataframe(filtered_data[showData], use_container_width=True)  
    
    # Graphs with enhanced styling
    # Graph 1: Yearly Catch Data
    tangkapan_tahunan = filtered_data.groupby('tahun').agg({'berat': 'sum'}).reset_index()
    fig_tangkapan = px.line(
        tangkapan_tahunan, 
        x='tahun', 
        y='berat', 
        title='TOTAL BERAT TANGKAPAN PER TAHUN',
        markers=True,
        color_discrete_sequence=['#1E88E5'],
    )
    fig_tangkapan.update_layout(
        xaxis=dict(tickmode='linear'),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(showgrid=True, gridcolor='#EEEEEE'),
        xaxis_title="Tahun",
        yaxis_title="Berat (Kg)",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=16, color='#333'),
        paper_bgcolor='white',
        hovermode='x unified',
    )
    fig_tangkapan.update_traces(
        line=dict(width=3),
        marker=dict(size=8),
    )

    # Graph 2: Top 10 Fish Types
    tangkapan_dominan = (
        filtered_data.groupby('nama_ikan_id').agg({'berat': 'sum'})
        .reset_index().sort_values(by='berat', ascending=False).head(10)
    )
    fig_tangkapan_dominan = px.bar(
        tangkapan_dominan,
        x='berat', 
        y='nama_ikan_id',
        orientation='h',
        title="JENIS TANGKAPAN TERBANYAK",
        color_discrete_sequence=['#1E88E5'],
    )
    fig_tangkapan_dominan.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True, 
            gridcolor='#EEEEEE',
            categoryorder='total ascending',
            title="Jenis Ikan",
        ),
        xaxis=dict(
            showgrid=True, 
            gridcolor='#EEEEEE',
            title="Berat (Kg)",
        ),
        height=400,
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=16, color='#333'),
        paper_bgcolor='white',
    )
    fig_tangkapan_dominan.update_traces(
        marker_color='#1E88E5',
        hovertemplate='<b>%{y}</b><br>Berat: %{x:,.0f} Kg<extra></extra>'
    )
    
    # Pie chart with better styling
    alat_tangkap_dominan = filtered_data.groupby('jenis_api').agg({'berat': 'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10)
    fig_alat_tangkap = px.pie(
        alat_tangkap_dominan, 
        names='jenis_api', 
        values='berat', 
        title='ALAT TANGKAP DOMINAN',
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig_alat_tangkap.update_layout(
        legend_title='Alat Tangkap',
        legend_y=0.9,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        title_font=dict(size=16, color='#333'),
        paper_bgcolor='white',
    )
    fig_alat_tangkap.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+percent+value',
        textfont_size=12,
    )

    # Display charts in a more visually appealing way
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(fig_tangkapan, use_container_width=True)
    with right:
        st.plotly_chart(fig_tangkapan_dominan, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Display pie chart in its own container
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.plotly_chart(fig_alat_tangkap, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
          

elif menu == 'Analysis':
    # Header
    st.markdown("""
    <div style="background-color:#1E3A8A; padding:10px; border-radius:10px; margin-bottom:20px;">
        <h1 style="color:white; text-align:center;">📊 Analysis Dashboard</h1>
        <p style="color:#E5E7EB; text-align:center; font-size:1.2em;">Analisis data perikanan menggunakan model statistik dan visualisasi</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload dan download
    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div style="background-color:#F3F4F6; padding:15px; border-radius:5px; border-left:5px solid #1E40AF;">
                <h3 style="color:#1E3A8A;">Upload Data CSV</h3>
                <p style="color:#4B5563;">Unggah file CSV Anda untuk analisis data perikanan.</p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                'Pilih file CSV untuk dianalisis',
                type=['csv'],
                help='Limit: 200MB per file'
            )

        with col2:
            st.markdown("""
            <div style="background-color:#F3F4F6; padding:15px; border-radius:5px; border-left:5px solid #059669;">
                <h3 style="color:#065F46;">Download Sample Data</h3>
                <p style="color:#4B5563;">Download format CSV sampel.</p>
            </div>
            """, unsafe_allow_html=True)

            try:
                with open('./data/data_kembung_karangantu.csv', 'r') as file:
                    sample_csv_content = file.read()

                st.download_button(
                    label='📥 Download Sample CSV',
                    data=sample_csv_content,
                    file_name='sample_data.csv',
                    mime='text/csv',
                    help='Klik untuk mengunduh data sampel'
                )
            except FileNotFoundError:
                st.error('Sample data tidak ditemukan.')

    st.markdown("<hr style='border: 1px solid #E5E7EB; margin: 20px 0;'>", unsafe_allow_html=True)

    # Proses file yang diupload
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)

        st.markdown("""
        <div style="background-color:#ECFDF5; padding:15px; border-radius:5px; border:1px solid #10B981;">
            <h4 style="color:#065F46;">✓ File berhasil diupload!</h4>
            <p style="color:#065F46;">Data siap untuk dianalisis</p>
        </div>
        """, unsafe_allow_html=True)

        num_rows = user_data.shape[0]
        num_cols = user_data.shape[1]

        st.markdown(f"""
        <div style="background-color:#F9FAFB; padding:15px; border-radius:8px; margin:15px 0;">
            <h3 style="color:#1F2937;">Dataset Overview</h3>
            <div style="display:flex; gap:20px;">
                <div style="flex:1; background:white; padding:15px; border-left:4px solid #3B82F6;">
                    <h4 style="color:#3B82F6;">Jumlah Data</h4>
                    <p style="font-size:1.5em;">{num_rows:,}</p>
                </div>
                <div style="flex:1; background:white; padding:15px; border-left:4px solid #8B5CF6;">
                    <h4 style="color:#8B5CF6;">Jumlah Kolom</h4>
                    <p style="font-size:1.5em;">{num_cols}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander('📋 Lihat Dataset'):
            st.dataframe(user_data, height=300)

        # Contoh analisis
        if 'tahun' in user_data.columns:
            st.markdown("""
            <div style="background-color:#EFF6FF; padding:15px; border-radius:10px; margin:20px 0;">
                <h2 style="color:#1E3A8A; text-align:center;">Analisis Data Perikanan</h2>
                <p style="color:#4B5563; text-align:center;">Visualisasi data tangkapan dan alat tangkap</p>
            </div>
            """, unsafe_allow_html=True)


            if 'Nilai Produksi' in user_data.columns:
                # Mengelompokkan data berdasarkan tahun dan menjumlahkan berat
                data_per_year = user_data.groupby('tahun').agg({'berat': 'sum', 'Nilai Produksi': 'sum'}).reset_index()

                # Menghitung rata-rata nilai produksi dan nilai produksi
                data_per_year['Harga rata-rata nilai produksi'] = data_per_year['Nilai Produksi'] / data_per_year['berat']
                data_per_year['Produksi (Ton)'] = data_per_year['berat'] / 1000
                data_per_year['Nilai Produksi'] = data_per_year['Produksi (Ton)'] * data_per_year['Harga rata-rata nilai produksi'] 
            else:
                data_per_year = user_data.groupby('tahun').agg({'berat': 'sum'}).reset_index()
                data_per_year['Produksi (Ton)'] = data_per_year['berat'] / 1000
                data_per_year['Harga rata-rata nilai produksi'] = None
                data_per_year['Nilai Produksi'] = None
            
            # Membuat grafik garis untuk total berat tangkapan per tahun
            fig_data_per_year = px.line(
                data_per_year, 
                x='tahun', 
                y='berat', 
                orientation='v',
                title='Total Berat Tangkapan Per Tahun', 
                template='plotly_white'  # Ubah template jadi white untuk tampilan lebih bersih
            )
            
            # Memperbarui layout grafik
            fig_data_per_year.update_layout(
                title={
                    'text': '<b>TOTAL BERAT TANGKAPAN PER TAHUN</b>',
                    'font': {'size': 20, 'color': '#1E3A8A'},
                    'y': 0.95
                },
                xaxis=dict(
                    tickmode='linear',
                    title='Tahun',
                    title_font={'size': 14, 'color': '#4B5563'},
                    tickfont={'size': 12},
                    gridcolor='#E5E7EB'
                ),
                yaxis=dict(
                    title='Berat (kg)',
                    title_font={'size': 14, 'color': '#4B5563'},
                    tickfont={'size': 12},
                    gridcolor='#E5E7EB',
                    showgrid=True
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hovermode='x unified',
                margin=dict(l=60, r=40, t=80, b=60)
            )
            
            # Perbarui garis dengan warna dan marker
            fig_data_per_year.update_traces(
                line=dict(color='#2563EB', width=3),
                marker=dict(size=8, color='#1E40AF'),
                hovertemplate='<b>Tahun:</b> %{x}<br><b>Berat:</b> %{y:,.0f} kg<extra></extra>'
            )

            # Grafik jenis API dominan
            api_dominan = user_data.groupby('jenis_api').agg({'berat':'sum'}).reset_index().sort_values(by='berat', ascending=False).head(10)
            
            # Mengubah tampilan grafik bar
            fig_api_dominan = px.bar(
                api_dominan,
                x='berat', 
                y='jenis_api',
                orientation='h',
                title="Jenis API Dominan",
                template='plotly_white',  # Menggunakan template putih
                color_discrete_sequence=['#3B82F6', '#60A5FA', '#93C5FD', '#BFDBFE', '#DBEAFE'] * 2  # Palet warna biru gradient
            )
            
            fig_api_dominan.update_layout(
                title={
                    'text': '<b>JENIS API DOMINAN</b>',
                    'font': {'size': 20, 'color': '#1E3A8A'},
                    'y': 0.95
                },
                xaxis=dict(
                    title='Berat (kg)',
                    title_font={'size': 14, 'color': '#4B5563'},
                    tickfont={'size': 12},
                    gridcolor='#E5E7EB',
                    showgrid=True
                ),
                yaxis=dict(
                    title='Jenis API',
                    title_font={'size': 14, 'color': '#4B5563'},
                    tickfont={'size': 12},
                    categoryorder='total ascending',
                    gridcolor='#E5E7EB'
                ),
                plot_bgcolor='white',
                paper_bgcolor='white',
                hoverlabel=dict(bgcolor='white', font_size=12),
                margin=dict(l=60, r=40, t=80, b=60)
            )
            
            # Update traces untuk custom hover
            fig_api_dominan.update_traces(
                hovertemplate='<b>%{y}</b><br>Berat: %{x:,.0f} kg<extra></extra>'
            )
            
            # Menampilkan grafik dalam layout yang lebih baik
            st.markdown("""
            <style>
            .chart-container {
                background-color: #F9FAFB;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display charts in columns
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            left, right = st.columns(2)
            with left:
                st.plotly_chart(fig_data_per_year, use_container_width=True)
            with right:
                st.plotly_chart(fig_api_dominan, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Tampilkan data produksi dalam card yang lebih baik
            st.markdown("""
            <div style="background-color:#F0FDF4; padding:15px; border-radius:10px; margin:15px 0; border:1px solid #D1FAE5;">
                <h3 style="color:#065F46; text-align:center; margin-bottom:15px;">📊 DATA PRODUKSI DAN NILAI PRODUKSI PER TAHUN</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Style DataTable
            st.markdown("""
            <style>
                .dataframe {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                }
                .dataframe th {
                    background-color: #065F46;
                    color: white;
                    padding: 12px;
                    text-align: left;
                }
                .dataframe td {
                    padding: 10px;
                    border-bottom: 1px solid #D1FAE5;
                }
                .dataframe tr:nth-child(even) {
                    background-color: #ECFDF5;
                }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                data_per_year[['tahun', 'Produksi (Ton)', 'Harga rata-rata nilai produksi', 'Nilai Produksi']]
                .reset_index(drop=True), 
                use_container_width=True
            )
            
        

# Ganti bagian expander dengan sistem tab yang lebih elegan
# Tambahkan custom CSS untuk tab dan perbaikan tampilan visual
        st.markdown("""
        <style>
            /* Styling untuk tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
                background-color: #F3F4F6;
                border-radius: 10px 10px 0px 0px;
                padding: 5px 5px 0px 5px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 45px;
                white-space: pre-wrap;
                background-color: #F3F4F6;
                border-radius: 10px 10px 0px 0px;
                gap: 1px;
                padding-left: 20px;
                padding-right: 20px;
                font-weight: 600;
                color: #4B5563;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1E40AF;
                color: white;
            }
            .stTabs [data-baseweb="tab-highlight"] {
                display: none;
            }
            /* Content container styling */
            .tab-content {
                background-color: white;
                border-radius: 0px 0px 10px 10px;
                padding: 20px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            }
            /* Charts container */
            .chart-container {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                margin-top: 15px;
                border: 1px solid #E5E7EB;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            /* Table styling */
            .styled-table {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                border: 1px solid #E5E7EB;
                margin-bottom: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            /* Header styling */
            .header-section {
                background-color: #EFF6FF;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 25px;
                border-left: 5px solid #1E40AF;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .header-title {
                color: #1E40AF;
                font-size: 1.6em;
                margin-bottom: 8px;
                font-weight: bold;
            }
            .header-subtitle {
                color: #4B5563;
                font-size: 1.1em;
                line-height: 1.4;
            }
            /* Tabel dataframe styling */
            .dataframe-container {
                padding: 0px !important;
                max-height: 350px;
                overflow-y: auto;
            }
            .dataframe-container td, .dataframe-container th {
                font-size: 13px !important;
                padding: 5px 10px !important;
            }
            /* Tombol filter section */
            .filter-section {
                background-color: #F9FAFB;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #E5E7EB;
            }
            /* Card summary styling */
            .summary-card {
                background-color: #F0F9FF;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                border-left: 4px solid #0284C7;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            }
            .summary-title {
                color: #0C4A6E;
                font-size: 1.1em;
                font-weight: 600;
                margin-bottom: 5px;
            }
            .summary-value {
                color: #0369A1;
                font-size: 1.8em;
                font-weight: 700;
            }
            .summary-label {
                color: #64748B;
                font-size: 0.85em;
            }
            /* Top 5 alat tangkap styling - untuk mengurangi jumlah tampilan di chart */
            .top-note {
                color: #64748B;
                font-size: 0.85em;
                font-style: italic;
                margin-top: 5px;
                text-align: center;
            }
        </style>
        """, unsafe_allow_html=True)

        # Header section untuk analisis alat tangkap
        st.markdown("""
        <div class="header-section">
            <div class="header-title">Analisis Alat Tangkap</div>
            <div class="header-subtitle">Data hasil tangkapan dan jumlah trip berdasarkan jenis alat penangkapan ikan</div>
        </div>
        """, unsafe_allow_html=True)

        # Membuat tabs untuk hasil tangkapan dan jumlah trip
        tab_hasil, tab_trip = st.tabs(["📊 Hasil Tangkapan per Alat", "🚢 Jumlah Trip per Alat"])

        # Tab 1: Hasil Tangkapan per Alat Tangkap
        with tab_hasil:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            if {'jenis_api', 'tahun', 'berat'}.issubset(user_data.columns):
                # Tambahkan section filter tahun (optional)
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
                with col_filter1:
                    tahun_list = sorted(user_data['tahun'].unique().tolist())
                    selected_years = st.multiselect("Pilih Tahun", tahun_list, default=tahun_list)
                
                with col_filter2:
                    # Filter untuk top alat tangkap
                    show_top = st.checkbox("Tampilkan 5 Alat Tangkap Teratas", value=True)
                
                # Jika tidak ada tahun yang dipilih, gunakan semua tahun
                if not selected_years:
                    selected_years = tahun_list
                    
                # Filter data berdasarkan tahun yang dipilih
                filtered_data = user_data[user_data['tahun'].isin(selected_years)]
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Hitung summary card untuk hasil tangkapan
                total_tangkapan = filtered_data['berat'].sum()
                avg_per_trip = total_tangkapan / filtered_data['Jumlah Hari'].sum() if filtered_data['Jumlah Hari'].sum() > 0 else 0
                
                # Tampilkan summary cards
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.markdown(f"""
                    <div class="summary-card">
                        <div class="summary-title">Total Hasil Tangkapan</div>
                        <div class="summary-value">{total_tangkapan:,.1f} kg</div>
                        <div class="summary-label">Dari {len(selected_years)} tahun terpilih</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum2:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #7E22CE;">
                        <div class="summary-title" style="color: #581C87;">Rata-rata per Trip</div>
                        <div class="summary-value" style="color: #7E22CE;">{avg_per_trip:,.1f} kg</div>
                        <div class="summary-label">Per hari trip</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum3:
                    unique_alat = filtered_data['jenis_api'].nunique()
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #16A34A;">
                        <div class="summary-title" style="color: #166534;">Jumlah Jenis Alat</div>
                        <div class="summary-value" style="color: #16A34A;">{unique_alat}</div>
                        <div class="summary-label">Jenis alat tangkap digunakan</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Layout dengan dua kolom
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown('<div class="styled-table">', unsafe_allow_html=True)
                    tangkapan_per_tahun = filtered_data.groupby(['jenis_api', 'tahun']).agg({'berat': 'sum'}).reset_index()
                    tangkapan_pivot = tangkapan_per_tahun.pivot(index='jenis_api', columns='tahun', values='berat').fillna(0)

                    # Tambahkan kolom total untuk tiap alat tangkap
                    tangkapan_pivot['Total'] = tangkapan_pivot.sum(axis=1)
                    
                    # Sort berdasarkan total (descending)
                    tangkapan_pivot = tangkapan_pivot.sort_values(by='Total', ascending=False)

                    # Tambahkan baris jumlah total untuk tiap alat tangkap
                    tangkapan_pivot.loc['Jumlah'] = tangkapan_pivot.sum()

                    # Reset index untuk tampilkan tabel
                    tangkapan_pivot = tangkapan_pivot.reset_index()
                    
                    # Format angka dengan pemisah ribuan
                    for col in tangkapan_pivot.columns:
                        if col != 'jenis_api':
                            tangkapan_pivot[col] = tangkapan_pivot[col].apply(lambda x: f"{x:,.1f}" if isinstance(x, (int, float)) else x)

                    # Tampilkan tabel dengan judul
                    st.markdown('<h3 style="color:#1E40AF; font-size:1.2em;">Hasil Tangkapan per Alat Tangkap (kg)</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(tangkapan_pivot, use_container_width=True, height=300)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Filter data untuk grafik (tanpa baris Jumlah)
                    chart_data = tangkapan_pivot[tangkapan_pivot['jenis_api'] != 'Jumlah'].copy()
                    
                    # Ambil top 5 jika opsi dipilih
                    if show_top and len(chart_data) > 5:
                        chart_data = chart_data.sort_values(by='Total', ascending=False).head(5)
                        top_note = True
                    else:
                        top_note = False
                    
                    # Konversi string ke float untuk kolom Total
                    chart_data['Total_num'] = chart_data['Total'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
                    
                    # Membuat grafik batang hasil tangkapan dengan tampilan lebih baik
                    fig_tangkapan_total = px.bar(
                        chart_data,
                        x='jenis_api',
                        y='Total_num',
                        title="<b>Hasil Tangkapan per Alat Tangkap</b>",
                        color='jenis_api',
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        labels={'jenis_api': 'Jenis Alat', 'Total_num': 'Total Tangkapan (kg)'},
                        template='plotly_white'
                    )
                    
                    fig_tangkapan_total.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.5)',
                        paper_bgcolor='rgba(255,255,255,0)',
                        title_font=dict(size=18, color='#1E40AF'),
                        legend_title_text='',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)',
                            tickformat=",.0f"
                        ),
                        margin=dict(l=50, r=30, t=80, b=50),
                    )
                    
                    # Tambahkan label nilai di atas setiap batang
                    fig_tangkapan_total.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Total: %{y:,.1f} kg<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_tangkapan_total, use_container_width=True)
                    
                    # Tampilkan catatan jika hanya menampilkan top 5
                    if top_note:
                        st.markdown('<div class="top-note">* Menampilkan 5 alat tangkap dengan hasil tertinggi</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tambahkan grafik pie chart untuk distribusi
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    # Gabungkan kategori kecil jika terlalu banyak
                    pie_data = chart_data.copy()
                    if len(pie_data) > 7 and not show_top:
                        # Ambil top 6, gabungkan sisanya sebagai "Lainnya"
                        pie_top = pie_data.nlargest(6, 'Total_num')
                        pie_others = pd.DataFrame({
                            'jenis_api': ['Lainnya'],
                            'Total_num': [pie_data['Total_num'].sum() - pie_top['Total_num'].sum()]
                        })
                        pie_data = pd.concat([pie_top, pie_others])
                    
                    fig_pie = px.pie(
                        pie_data,
                        values='Total_num',
                        names='jenis_api',
                        title='<b>Distribusi Hasil Tangkapan (%)</b>',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    
                    fig_pie.update_layout(
                        title_font=dict(size=18, color='#1E40AF'),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=80, b=20),
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>%{value:,.1f} kg<br>%{percent}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Data tidak memiliki semua kolom yang diperlukan (jenis_api, tahun, berat).")
                
            st.markdown('</div>', unsafe_allow_html=True)

        # Tab 2: Jumlah Trip per Alat Tangkap
        with tab_trip:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            
            if {'jenis_api', 'tahun', 'Jumlah Hari'}.issubset(user_data.columns):
                # Tambahkan section filter tahun (optional)
                st.markdown('<div class="filter-section">', unsafe_allow_html=True)
                col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
                with col_filter1:
                    tahun_list = sorted(user_data['tahun'].unique().tolist())
                    selected_years_trip = st.multiselect("Pilih Tahun", tahun_list, default=tahun_list, key="trip_years")
                
                with col_filter2:
                    # Filter untuk top alat tangkap
                    show_top_trip = st.checkbox("Tampilkan 5 Alat Tangkap Teratas", value=True, key="top_trip")
                
                # Jika tidak ada tahun yang dipilih, gunakan semua tahun
                if not selected_years_trip:
                    selected_years_trip = tahun_list
                    
                # Filter data berdasarkan tahun yang dipilih
                filtered_data_trip = user_data[user_data['tahun'].isin(selected_years_trip)]
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Hitung summary card untuk jumlah trip
                total_trip = filtered_data_trip['Jumlah Hari'].sum()
                
                # Hitung rata-rata hasil per hari
                if 'berat' in filtered_data_trip.columns:
                    total_tangkapan_trip = filtered_data_trip['berat'].sum()
                    avg_tangkapan_per_trip = total_tangkapan_trip / total_trip if total_trip > 0 else 0
                else:
                    avg_tangkapan_per_trip = 0
                
                # Tampilkan summary cards
                col_sum1, col_sum2, col_sum3 = st.columns(3)
                with col_sum1:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #EA580C;">
                        <div class="summary-title" style="color: #9A3412;">Total Hari Trip</div>
                        <div class="summary-value" style="color: #EA580C;">{int(total_trip):,}</div>
                        <div class="summary-label">Hari</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum2:
                    # Hitung jumlah total trip berbeda (perjalanan, bukan hari)
                    jumlah_trip_unik = len(filtered_data_trip)
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #0891B2;">
                        <div class="summary-title" style="color: #155E75;">Jumlah Trip Tercatat</div>
                        <div class="summary-value" style="color: #0891B2;">{jumlah_trip_unik:,}</div>
                        <div class="summary-label">Perjalanan</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_sum3:
                    st.markdown(f"""
                    <div class="summary-card" style="border-left-color: #4F46E5;">
                        <div class="summary-title" style="color: #3730A3;">Rata-rata Hasil per Hari</div>
                        <div class="summary-value" style="color: #4F46E5;">{avg_tangkapan_per_trip:,.1f} kg</div>
                        <div class="summary-label">Hasil tangkapan per hari trip</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Layout dengan dua kolom
                col1, col2 = st.columns([2, 3])
                
                with col1:
                    st.markdown('<div class="styled-table">', unsafe_allow_html=True)
                    effort_per_tahun = filtered_data_trip.groupby(['jenis_api', 'tahun']).agg({'Jumlah Hari': 'sum'}).reset_index()
                    effort_pivot = effort_per_tahun.pivot(index='jenis_api', columns='tahun', values='Jumlah Hari').fillna(0)

                    # Tambahkan kolom total untuk tiap alat tangkap
                    effort_pivot['Total'] = effort_pivot.sum(axis=1)
                    
                    # Sort berdasarkan total (descending)
                    effort_pivot = effort_pivot.sort_values(by='Total', ascending=False)
                    
                    # Tambahkan baris jumlah total untuk tiap alat tangkap
                    effort_pivot.loc['Jumlah'] = effort_pivot.sum()

                    # Reset index untuk tampilkan tabel
                    effort_pivot = effort_pivot.reset_index()
                    
                    # Format angka dengan pemisah ribuan
                    for col in effort_pivot.columns:
                        if col != 'jenis_api':
                            effort_pivot[col] = effort_pivot[col].apply(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                    
                    # Tampilkan tabel dengan judul
                    st.markdown('<h3 style="color:#1E40AF; font-size:1.2em;">Jumlah Trip per Alat Tangkap (hari)</h3>', unsafe_allow_html=True)
                    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                    st.dataframe(effort_pivot, use_container_width=True, height=300)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    # Filter data untuk grafik (tanpa baris Jumlah)
                    chart_data = effort_pivot[effort_pivot['jenis_api'] != 'Jumlah'].copy()
                    
                    # Ambil top 5 jika opsi dipilih
                    if show_top_trip and len(chart_data) > 5:
                        chart_data = chart_data.sort_values(by='Total', ascending=False).head(5)
                        top_trip_note = True
                    else:
                        top_trip_note = False
                    
                    # Konversi string ke float untuk kolom Total
                    chart_data['Total_num'] = chart_data['Total'].apply(lambda x: float(x.replace(',', '')) if isinstance(x, str) else x)
                    
                    # Membuat grafik batang jumlah trip dengan tampilan lebih baik
                    fig_trip_per_alat = px.bar(
                        chart_data,
                        x='jenis_api',
                        y='Total_num',
                        title="<b>Jumlah Trip per Alat Tangkap</b>",
                        color='jenis_api',
                        color_discrete_sequence=px.colors.qualitative.Vivid,
                        labels={'jenis_api': 'Jenis Alat', 'Total_num': 'Total Trip (hari)'},
                        template='plotly_white'
                    )
                    
                    fig_trip_per_alat.update_layout(
                        plot_bgcolor='rgba(255,255,255,0.5)',
                        paper_bgcolor='rgba(255,255,255,0)',
                        title_font=dict(size=18, color='#1E40AF'),
                        legend_title_text='',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)'
                        ),
                        yaxis=dict(
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            showgrid=True,
                            gridcolor='rgba(230,230,230,0.8)',
                            tickformat=",.0f"
                        ),
                        margin=dict(l=50, r=30, t=80, b=50),
                    )
                    
                    # Tambahkan label nilai di atas setiap batang
                    fig_trip_per_alat.update_traces(
                        texttemplate='%{y:,.0f}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Total: %{y:,.0f} hari<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_trip_per_alat, use_container_width=True)
                    
                    # Tampilkan catatan jika hanya menampilkan top 5
                    if top_trip_note:
                        st.markdown('<div class="top-note">* Menampilkan 5 alat tangkap dengan jumlah trip tertinggi</div>', unsafe_allow_html=True)
                        
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Tambahkan grafik pie chart untuk distribusi
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    
                    # Gabungkan kategori kecil jika terlalu banyak
                    pie_data_trip = chart_data.copy()
                    if len(pie_data_trip) > 7 and not show_top_trip:
                        # Ambil top 6, gabungkan sisanya sebagai "Lainnya"
                        pie_top = pie_data_trip.nlargest(6, 'Total_num')
                        pie_others = pd.DataFrame({
                            'jenis_api': ['Lainnya'],
                            'Total_num': [pie_data_trip['Total_num'].sum() - pie_top['Total_num'].sum()]
                        })
                        pie_data_trip = pd.concat([pie_top, pie_others])
                    
                    fig_pie = px.pie(
                        pie_data_trip,
                        values='Total_num',
                        names='jenis_api',
                        title='<b>Distribusi Jumlah Trip (%)</b>',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Vivid
                    )
                    
                    fig_pie.update_layout(
                        title_font=dict(size=18, color='#1E40AF'),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                        margin=dict(l=20, r=20, t=80, b=20),
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>%{value:,.0f} hari<br>%{percent}<extra></extra>'
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Data tidak memiliki semua kolom yang diperlukan (jenis_api, tahun, Jumlah Hari).")
            
            st.markdown('</div>', unsafe_allow_html=True)

       
            
            st.markdown('</div>', unsafe_allow_html=True)

                # Header untuk analisis lanjutan dengan desain modern
        st.markdown("""
<div style="background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%); 
     padding: 25px; border-radius: 12px; margin: 30px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="background-color: rgba(255,255,255,0.2); color: white; padding: 12px; 
              border-radius: 50%; margin-right: 15px; font-size: 24px;">📈</span>
        <h2 style="color: white; margin: 0; font-weight: 600; font-size: 28px;">ANALISIS LANJUTAN: MODEL PRODUKSI SURPLUS</h2>
    </div>
    <p style="color: rgba(255,255,255,0.9); margin: 0; padding-left: 60px; font-size: 16px; line-height: 1.5;">
        Model Produksi Surplus adalah metode analisis stok ikan yang mengukur hubungan antara kelimpahan stok 
        dan upaya penangkapan. Model ini membantu dalam menentukan tingkat eksploitasi optimal dan berperan 
        penting dalam menjaga keberlanjutan sumber daya perikanan.
    </p>
</div>

<style>
    /* Perbaikan tampilan sidebar */
    [data-testid=stSidebar] {
        background-color: #F1F5F9;
    }
    
    /* Styling untuk bagian tab/expander */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3B82F6 !important;
        color: white !important;
    }
    
    /* Styling untuk expander */
    .streamlit-expanderHeader {
        background-color: #F1F5F9;
        border-radius: 8px;
        font-weight: 600;
        color: #1E3A8A;
    }
    
    .streamlit-expanderContent {
        background-color: white;
        border-radius: 0px 0px 8px 8px;
        border: 1px solid #E2E8F0;
        border-top: none;
    }
</style>
""", unsafe_allow_html=True)

            
        with st.expander('📊 CPUE Analysis', expanded=True):
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;'>
                <h3 style='color: #2E7D32;'>Catch Per Unit Effort (CPUE) Analysis</h3>
                <p>CPUE adalah rasio antara jumlah tangkapan ikan dengan upaya penangkapan yang dilakukan.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if 'jenis_api' in user_data.columns and 'berat' in user_data.columns and 'Jumlah Hari' in user_data.columns:
                # Standardize column names
                user_data.rename(columns={
                    'jenis_api': 'Alat Tangkap', 
                    'berat': 'catch (ton)', 
                    'Jumlah Hari': 'effort (hari)'
                }, inplace=True)

                # Convert to tons
                user_data['catch (ton)'] = user_data['catch (ton)'] / 1000 
                
                # Group data by gear type
                alat_tangkap_group = user_data.groupby('Alat Tangkap').agg({
                    'catch (ton)': 'sum', 
                    'effort (hari)': 'sum'
                }).reset_index()

                # Calculate CPUE
                alat_tangkap_group['CPUE'] = alat_tangkap_group['catch (ton)'] / alat_tangkap_group['effort (hari)']

                # Calculate contribution percentage
                total_catch = alat_tangkap_group['catch (ton)'].sum()
                alat_tangkap_group['percentage'] = (alat_tangkap_group['catch (ton)'] / total_catch) * 100

                # Sort by percentage
                alat_tangkap_group = alat_tangkap_group.sort_values(by='percentage', ascending=False)

                # Define thresholds
                DOMINANCE_THRESHOLD = 50
                SIGNIFICANT_THRESHOLD = 20

                # Check for dominant gear
                dominant_gear = alat_tangkap_group.iloc[0]
                is_dominant = dominant_gear['percentage'] >= DOMINANCE_THRESHOLD

                # Create columns for better layout
                col1, col2 = st.columns([1, 1])

                with col1:
                    if is_dominant:
                        st.info(f"**Alat tangkap dominan: {dominant_gear['Alat Tangkap']}** dengan persentase tangkapan **{dominant_gear['percentage']:.2f}%** dari total tangkapan")
                        alat_tangkap_dominan = pd.DataFrame([dominant_gear])
                        alat_tangkap_dominan['FPI'] = 1.0
                    else:
                        significant_gear = alat_tangkap_group[alat_tangkap_group['percentage'] >= SIGNIFICANT_THRESHOLD]
                        st.warning(f"**Tidak ada alat tangkap dominan (≥50%)**. Melakukan standarisasi untuk **{len(significant_gear)}** alat tangkap yang berkontribusi ≥20%.")
                        alat_tangkap_dominan = significant_gear.copy()
                        cpue_max = alat_tangkap_dominan['CPUE'].max()
                        alat_tangkap_dominan['FPI'] = alat_tangkap_dominan['CPUE'] / cpue_max
                        alat_tangkap_dominan.loc[alat_tangkap_dominan['CPUE'] == cpue_max, 'FPI'] = 1

                # Display CPUE data table
                st.subheader('Data CPUE per Alat Tangkap')
                display_cols = ['Alat Tangkap', 'catch (ton)', 'effort (hari)', 'CPUE', 'percentage', 'FPI']
                
                # Style the dataframe
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['background-color: rgba(76, 175, 80, 0.2)' if v else '' for v in is_max]
                    
                styled_df = alat_tangkap_dominan[display_cols].style\
                    .format({'catch (ton)': '{:.2f}', 'effort (hari)': '{:.1f}', 'CPUE': '{:.4f}', 
                            'percentage': '{:.2f}%', 'FPI': '{:.3f}'})\
                    .apply(highlight_max, subset=['CPUE', 'percentage'])
                    
                st.dataframe(styled_df)

                # Create CPUE bar chart
                fig_cpue = px.bar(
                    alat_tangkap_dominan,
                    x='Alat Tangkap',
                    y='CPUE',
                    text='CPUE',    
                    title='CPUE per Alat Tangkap',
                    labels={'Alat Tangkap': 'Jenis Alat Tangkap', 'CPUE': 'CPUE (ton/hari)'},
                    template='plotly_white',
                    color='Alat Tangkap',
                    color_discrete_sequence=px.colors.qualitative.G10
                )
                fig_cpue.update_traces(texttemplate='%{text:.4f}', textposition='outside')
                fig_cpue.update_layout(
                    showlegend=False,
                    xaxis_title_font={'size': 14},
                    yaxis_title_font={'size': 14},
                    title_font={'size': 16},
                    plot_bgcolor='rgba(248,249,250,1)'
                )

                st.plotly_chart(fig_cpue, use_container_width=True)

                # Calculate yearly data
                if is_dominant:
                    yearly_effort = user_data[user_data['Alat Tangkap'] == dominant_gear['Alat Tangkap']].groupby('tahun')['effort (hari)'].sum().reset_index()
                else:
                    standardized_efforts = []
                    for gear in alat_tangkap_dominan['Alat Tangkap']:
                        gear_data = user_data[user_data['Alat Tangkap'] == gear]
                        gear_fpi = alat_tangkap_dominan.loc[alat_tangkap_dominan['Alat Tangkap'] == gear, 'FPI'].iloc[0]

                        gear_effort = gear_data.groupby('tahun')['effort (hari)'].sum() * gear_fpi
                        standardized_efforts.append(pd.DataFrame({
                            'tahun': gear_effort.index,
                            'effort_std': gear_effort.values
                        }))

                    combined_efforts = pd.concat(standardized_efforts)
                    yearly_effort = combined_efforts.groupby('tahun')['effort_std'].sum().reset_index()
                    yearly_effort.columns = ['tahun', 'effort (hari)']

                # Calculate yearly catch
                yearly_catch = user_data.groupby('tahun')['catch (ton)'].sum().reset_index()

                # Merge catch and effort data
                yearly_data = pd.merge(yearly_catch, yearly_effort, on='tahun')
                yearly_data['CPUE'] = yearly_data['catch (ton)'] / yearly_data['effort (hari)']

                # Display yearly data
                st.subheader('Data Tahunan')
                yearly_styled = yearly_data.style.format({
                    'catch (ton)': '{:.2f}', 
                    'effort (hari)': '{:.1f}', 
                    'CPUE': '{:.4f}'})
                st.dataframe(yearly_styled)

                # R-squared calculation function
                def calculate_r2(y_true, y_pred):
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                    r2 = 1 - (ss_res / ss_tot)
                    return r2

                # SCHAEFER MODEL
                def calculate_schaefer(data):
                    X = data['effort (hari)'].values.reshape(-1, 1)
                    Y = data['CPUE']

                    model = LinearRegression()
                    model.fit(X, Y)

                    a = model.intercept_
                    b = model.coef_[0]

                    Y_pred = model.predict(X)
                    r2 = calculate_r2(Y, Y_pred)

                    Eopt = -a / (2*b)
                    CMSY = -(a ** 2) / (4 * b)

                    return {
                        'name': 'Schaefer',
                        'a': a,
                        'b': b,
                        'Eopt': Eopt,
                        'CMSY': CMSY,
                        'R2': r2
                    }

                # FOX MODEL
                def calculate_fox(data):
                    X = data['effort (hari)'].values.reshape(-1, 1)
                    Y = np.log(data['CPUE'])

                    model = LinearRegression()
                    model.fit(X, Y)

                    c = model.intercept_
                    d = model.coef_[0]

                    Y_pred = model.predict(X)
                    r2 = calculate_r2(Y, Y_pred)

                    Eopt = -1 / d
                    CMSY = -(1/d) * np.exp(c-1)

                    # Create array for effort and catch values
                    n_points = 20
                    effort_points = np.zeros(n_points)
                    catch_points = np.zeros(n_points)

                    effort_points[0] = 0
                    catch_points[0] = 0

                    for i in range(1, n_points):
                        effort_points[i] = effort_points[i-1] + (Eopt * 0.1)
                        catch_points[i] = effort_points[i] * np.exp(c + d * effort_points[i])

                    return {
                        'name': 'Fox',
                        'c': c,
                        'd': d,
                        'Eopt': Eopt,
                        'CMSY': CMSY,
                        'R2': r2,
                        'effort_range': effort_points,
                        'catch_pred': catch_points
                    }

                # Calculate both models
                with st.spinner('Menghitung model surplus produksi...'):
                    schaefer_results = calculate_schaefer(yearly_data)
                    fox_results = calculate_fox(yearly_data)

                # Display model results
                st.markdown("""
                <div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; border-left: 5px solid #2196F3;'>
                    <h3 style='color: #0d47a1;'>Hasil Perhitungan Model Surplus Produksi</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create a better comparison table
                comparison_df = pd.DataFrame({
                    'Parameter': ['a/c', 'b/d', 'R²', 'E optimal', 'MSY'],
                    'Model Schaefer': [
                        f"{schaefer_results['a']:.6f}",
                        f"{schaefer_results['b']:.6f}",    
                        f"{schaefer_results['R2']:.4f}",
                        f"{schaefer_results['Eopt']:.2f} hari",
                        f"{schaefer_results['CMSY']:.2f} ton"
                    ],
                    'Model Fox': [
                        f"{fox_results['c']:.4f}",
                        f"{fox_results['d']:.4f}",
                        f"{fox_results['R2']:.4f}",
                        f"{fox_results['Eopt']:.2f} hari",
                        f"{fox_results['CMSY']:.2f} ton"
                    ]
                })
                
                # Display comparison table with better styling
                st.dataframe(comparison_df.set_index('Parameter').T.style.highlight_max(axis=0, color='#e3f2fd'))

                # Model selection with better UI
                col1, col2 = st.columns([1, 1])
                with col1:
                    schaefer_r2 = schaefer_results['R2']
                    fox_r2 = fox_results['R2']
                    
                    # Automatically pre-select the model with higher R²
                    default_index = 0 if schaefer_r2 > fox_r2 else 1
                    
                    selected_model = st.radio(
                        "Pilih model untuk visualisasi:",
                        [f"Model Schaefer (R² = {schaefer_r2:.4f})",
                        f"Model Fox (R² = {fox_r2:.4f})"],
                        index=default_index
                    )
                    
                with col2:
                    if 'Schaefer' in selected_model:
                        st.success(f"**MSY: {schaefer_results['CMSY']:.2f} ton** pada effort optimal **{schaefer_results['Eopt']:.2f} hari**")
                    else:
                        st.success(f"**MSY: {fox_results['CMSY']:.2f} ton** pada effort optimal **{fox_results['Eopt']:.2f} hari**")

                # Generate visualization for selected model
                if 'Schaefer' in selected_model:
                    model_results = schaefer_results

                    # Create 20 data points from 0 to 2*Eopt
                    n_points = 50  # Increased for smoother curve
                    effort_range = np.linspace(0, 2 * model_results['Eopt'], n_points)
                    # Calculate catch using Schaefer model
                    catch_pred = effort_range * (model_results['a'] + model_results['b'] * effort_range)

                else:
                    model_results = fox_results
                    # Create data points according to book guide
                    n_points = 100
                    
                    # Initialize arrays for effort and catch
                    effort_range = np.zeros(n_points)
                    catch_pred = np.zeros(n_points)
                    
                    # First data point = 0
                    effort_range[0] = 0
                    catch_pred[0] = 0
                    
                    # Calculate effort and catch for points 2-100
                    for i in range(1, n_points):
                        effort_range[i] = effort_range[i-1] + (model_results['Eopt'] * 0.1)
                        catch_pred[i] = effort_range[i] * np.exp(model_results['c'] + model_results['d'] * effort_range[i])

                # Create dataframe for model predictions
                model_df = pd.DataFrame({
                    'Effort': effort_range,
                    'Catch': catch_pred
                })

                # Ensure catch value at first point = 0
                model_df.loc[0, 'Catch'] = 0

                # Create dataframe for actual data
                actual_df = yearly_data[['effort (hari)', 'catch (ton)']]

                # Create surplus production curve visualization
                fig_surplus = go.Figure()

                # Add model curve
                fig_surplus.add_trace(
                    go.Scatter(
                        x=model_df['Effort'],
                        y=model_df['Catch'],
                        mode='lines',
                        name=f"Model {model_results['name']}",
                        line=dict(color='#1E88E5', width=3)
                    )
                )

                # Add actual data points
                fig_surplus.add_trace(
                    go.Scatter(
                        x=actual_df['effort (hari)'],
                        y=actual_df['catch (ton)'],
                        mode='markers',
                        name='Data Aktual',
                        marker=dict(color='#D32F2F', size=10, line=dict(width=1, color='#7f7f7f'))
                    )
                )

                # Add MSY point
                fig_surplus.add_trace(
                    go.Scatter(
                        x=[model_results['Eopt']],
                        y=[model_results['CMSY']],
                        mode='markers',
                        name='MSY',
                        marker=dict(color='#4CAF50', size=14, symbol='star')
                    )
                )

                # Update layout with better styling
                fig_surplus.update_layout(
                    title=dict(
                        text=f"Hubungan Hasil Tangkapan dan Upaya Penangkapan (Model {model_results['name']})",
                        font=dict(size=18)
                    ),
                    xaxis_title=dict(text='Upaya Penangkapan (hari)', font=dict(size=14)),
                    yaxis_title=dict(text='Hasil Tangkapan (ton)', font=dict(size=14)),
                    template='plotly_white',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        orientation="h"
                    ),
                    # Ensure x and y axes start from 0
                    xaxis=dict(range=[0, max(effort_range)], zeroline=True, linewidth=2),
                    yaxis=dict(range=[0, max(catch_pred) * 1.2], zeroline=True, linewidth=2),
                    plot_bgcolor='rgba(248,249,250,1)',
                    hovermode='closest'
                )

                # Add annotation for MSY
                fig_surplus.add_annotation(
                    x=model_results['Eopt'],
                    y=model_results['CMSY'],
                    text=f"MSY = {model_results['CMSY']:.2f} ton<br>Eopt = {model_results['Eopt']:.2f} hari",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#4CAF50",
                    arrowsize=1,
                    arrowwidth=2,
                    ax=40,
                    ay=-40,
                    font=dict(color="#4CAF50", size=12),
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="#4CAF50",
                    borderwidth=1,
                    borderpad=4
                )

                st.plotly_chart(fig_surplus, use_container_width=True)

                # CPUE-Effort Relationship Graph
                st.markdown("""
                <div style='background-color: #fff3e0; padding: 10px; border-radius: 5px; border-left: 5px solid #FF9800;'>
                    <h3 style='color: #e65100;'>Hubungan CPUE dengan Effort</h3>
                </div>
                """, unsafe_allow_html=True)

                # Create new figure for CPUE-Effort relationship
                if 'Schaefer' in selected_model:
                    # Calculate predicted CPUE for Schaefer model
                    cpue_pred = model_results['a'] + model_results['b'] * effort_range
                    
                    fig_cpue = go.Figure()
                    
                    # Add regression line
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=effort_range,
                            y=cpue_pred,
                            mode='lines',
                            name='Model Schaefer',
                            line=dict(color='#1E88E5', width=3)
                        )
                    )
                    
                    # Add actual data points
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=yearly_data['effort (hari)'],
                            y=yearly_data['CPUE'],
                            mode='markers',
                            name='Data Aktual',
                            marker=dict(color='#D32F2F', size=10, line=dict(width=1, color='#7f7f7f'))
                        )
                    )
                    
                    # Update layout
                    fig_cpue.update_layout(
                        title=dict(text='Hubungan CPUE dengan Effort (Model Schaefer)', font=dict(size=18)),
                        xaxis_title=dict(text='Upaya Penangkapan (hari)', font=dict(size=14)),
                        yaxis_title=dict(text='CPUE (ton/hari)', font=dict(size=14)),
                        template='plotly_white',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            orientation="h"
                        ),
                        xaxis=dict(zeroline=True, linewidth=2),
                        yaxis=dict(zeroline=True, linewidth=2),
                        plot_bgcolor='rgba(248,249,250,1)',
                        hovermode='closest'
                    )
                    
                    # Add annotation for the formula
                    fig_cpue.add_annotation(
                        x=0.5,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"CPUE = {model_results['a']:.6f} + ({model_results['b']:.6f} × E)",
                        showarrow=False,
                        font=dict(size=14),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#1E88E5",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )
                    
                else:
                    # Calculate ln(CPUE) predictions for Fox model
                    ln_cpue_pred = model_results['c'] + model_results['d'] * effort_range
                    
                    fig_cpue = go.Figure()
                    
                    # Add regression line
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=effort_range,
                            y=ln_cpue_pred,
                            mode='lines',
                            name='Model Fox',
                            line=dict(color='#1E88E5', width=3)
                        )
                    )
                    
                    # Add actual data points
                    fig_cpue.add_trace(
                        go.Scatter(
                            x=yearly_data['effort (hari)'],
                            y=np.log(yearly_data['CPUE']),  # Using ln(CPUE) for actual data
                            mode='markers',
                            name='Data Aktual',
                            marker=dict(color='#D32F2F', size=10, line=dict(width=1, color='#7f7f7f'))
                        )
                    )
                    
                    # Update layout
                    fig_cpue.update_layout(
                        title=dict(text='Hubungan ln(CPUE) dengan Effort (Model Fox)', font=dict(size=18)),
                        xaxis_title=dict(text='Upaya Penangkapan (hari)', font=dict(size=14)),
                        yaxis_title=dict(text='ln(CPUE)', font=dict(size=14)),
                        template='plotly_white',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=-0.2,
                            xanchor="center",
                            x=0.5,
                            orientation="h"
                        ),
                        xaxis=dict(zeroline=True, linewidth=2),
                        yaxis=dict(zeroline=True, linewidth=2),
                        plot_bgcolor='rgba(248,249,250,1)',
                        hovermode='closest'
                    )
                    
                    # Add annotation for the formula
                    fig_cpue.add_annotation(
                        x=0.5,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        text=f"ln(CPUE) = {model_results['c']:.4f} + ({model_results['d']:.4f} × E)",
                        showarrow=False,
                        font=dict(size=14),
                        bgcolor="rgba(255, 255, 255, 0.8)",
                        bordercolor="#1E88E5",
                        borderwidth=1,
                        borderpad=4,
                        align="center"
                    )

                # Display CPUE-Effort graph
                st.plotly_chart(fig_cpue, use_container_width=True)

                # Data used in graphs - MOVED OUTSIDE THE EXPANDER
                st.subheader("Data yang Digunakan dalam Grafik")
                if 'Schaefer' in selected_model:
                    plot_data = pd.DataFrame({
                        'Effort': yearly_data['effort (hari)'],
                        'CPUE': yearly_data['CPUE'],
                        'CPUE_predicted': model_results['a'] + model_results['b'] * yearly_data['effort (hari)']
                    })
                    st.write('Model Schaefer:')
                else:
                    plot_data = pd.DataFrame({
                        'Effort': yearly_data['effort (hari)'],
                        'ln(CPUE)': np.log(yearly_data['CPUE']),
                        'ln(CPUE)_predicted': model_results['c'] + model_results['d'] * yearly_data['effort (hari)']
                    })
                    st.write('Model Fox:')

                st.dataframe(plot_data.round(4).style.highlight_max(axis=0))

                # Important points from the model
                st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;'>
                    <h3 style='color: #2E7D32;'>Titik-titik penting Model {model_results['name']}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                important_points = pd.DataFrame({
                    'Effort': [0, model_results['Eopt']/2, model_results['Eopt'], model_results['Eopt']*1.5, model_results['Eopt']*2],
                    'Catch': [0, model_results['CMSY']*0.75, model_results['CMSY'], 
                                model_results['CMSY']*0.75, 0]
                }).round(2)
                important_points['% dari MSY'] = (important_points['Catch'] / model_results['CMSY'] * 100).round(2)
                
                # Create a more visually appealing table for important points
                def highlight_optimal(s):
                    is_optimal = s == s.max()
                    return ['background-color: rgba(76, 175, 80, 0.2)' if v else '' for v in is_optimal]
                    
                styled_important = important_points.style\
                    .format({'Effort': '{:.2f} hari', 'Catch': '{:.2f} ton', '% dari MSY': '{:.2f}%'})\
                    .apply(highlight_optimal, subset=['Catch', '% dari MSY'])
                    
                st.dataframe(styled_important)
        


                # # Analisis Lanjutan: Model Produksi Surplus
        with st.expander("🔍 Evaluasi Keandalan Model", expanded=True):
            # Fungsi untuk menghitung metrik evaluasi
            def calculate_metrics(actual, predicted):
                """Menghitung metrik evaluasi model"""
                n = len(actual)
                
                # RMSE - Root Mean Square Error
                rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                
                # MAE - Mean Absolute Error
                mae = np.mean(np.abs(actual - predicted))
                
                # MAPE - Mean Absolute Percentage Error
                # Menghindari pembagian dengan nol
                actual_non_zero = np.where(actual != 0, actual, np.inf)
                mape = np.mean(np.abs((actual - predicted) / actual_non_zero)) * 100
                
                # Ganti nilai inf dengan NaN
                mape = np.nan if np.isinf(mape) else mape
                
                return {
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape
                }

            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Model Schaefer")
                # Prediksi catch menggunakan model Schaefer
                schaefer_catch_pred = yearly_data['effort (hari)'] * (schaefer_results['a'] + schaefer_results['b'] * yearly_data['effort (hari)'])
                schaefer_metrics = calculate_metrics(yearly_data['catch (ton)'], schaefer_catch_pred)
                
                st.metric(
                    label="R² Score", 
                    value=f"{schaefer_results['R2']:.4f}",
                    delta=None
                )
            
            with col2:
                st.subheader("Model Fox")
                # Prediksi catch menggunakan model Fox
                fox_catch_pred = yearly_data['effort (hari)'] * np.exp(fox_results['c'] + fox_results['d'] * yearly_data['effort (hari)'])
                fox_metrics = calculate_metrics(yearly_data['catch (ton)'], fox_catch_pred)
                
                st.metric(
                    label="R² Score", 
                    value=f"{fox_results['R2']:.4f}",
                    delta=None
                )

            # Bandingkan metrik evaluasi
            st.subheader("Perbandingan Metrik Evaluasi")
            metrics_df = pd.DataFrame({
                'Metrik': ['RMSE (ton)', 'MAE (ton)', 'MAPE (%)', 'R²'],
                'Schaefer': [
                    f"{schaefer_metrics['RMSE']:.3f}",
                    f"{schaefer_metrics['MAE']:.3f}",
                    f"{schaefer_metrics['MAPE']:.2f}" if not np.isnan(schaefer_metrics['MAPE']) else "N/A",
                    f"{schaefer_results['R2']:.4f}"
                ],
                'Fox': [
                    f"{fox_metrics['RMSE']:.3f}",
                    f"{fox_metrics['MAE']:.3f}",
                    f"{fox_metrics['MAPE']:.2f}" if not np.isnan(fox_metrics['MAPE']) else "N/A",
                    f"{fox_results['R2']:.4f}"
                ]
            })

            # Highlight best metrics
            def highlight_best(s):
                if s.name in ['RMSE (ton)', 'MAE (ton)', 'MAPE (%)']:
                    is_min = pd.Series(s.index.map(lambda x: float(s[x].split()[0]) if s[x] != "N/A" else float('inf')))
                    return ['background-color: rgba(152, 251, 152, 0.5)' if v == min(is_min) else '' for v in is_min]
                elif s.name == 'R²':
                    is_max = pd.Series(s.index.map(lambda x: float(s[x])))
                    return ['background-color: rgba(152, 251, 152, 0.5)' if v == max(is_max) else '' for v in is_max]
                return [''] * len(s)

            st.dataframe(metrics_df.set_index('Metrik').style.apply(highlight_best, axis=1), use_container_width=True)

            # Analisis residual
            st.subheader("📈 Analisis Residual", anchor=False)
            st.caption("Residual = Catch Aktual - Catch Prediksi")

            # Buat dataframe untuk analisis residual
            residual_df = pd.DataFrame({
                'Tahun': yearly_data['tahun'],
                'Catch Aktual (ton)': yearly_data['catch (ton)'],
                'Schaefer Pred. (ton)': schaefer_catch_pred,
                'Fox Pred. (ton)': fox_catch_pred,
                'Residual Schaefer': yearly_data['catch (ton)'] - schaefer_catch_pred,
                'Residual Fox': yearly_data['catch (ton)'] - fox_catch_pred
            })

            # Tabs untuk tabel dan visualisasi
            tab1, tab2 = st.tabs(["📊 Visualisasi Residual", "📋 Tabel Residual"])
            
            with tab1:
                # Visualisasi residual
                fig_residual = go.Figure()

                # Tambahkan residual untuk model Schaefer
                fig_residual.add_trace(
                    go.Bar(
                        x=yearly_data['tahun'],
                        y=residual_df['Residual Schaefer'],
                        name='Residual Schaefer',
                        marker_color='rgba(58, 71, 80, 0.8)'
                    )
                )

                # Tambahkan residual untuk model Fox
                fig_residual.add_trace(
                    go.Bar(
                        x=yearly_data['tahun'],
                        y=residual_df['Residual Fox'],
                        name='Residual Fox',
                        marker_color='rgba(246, 78, 139, 0.7)'
                    )
                )

                # Tambahkan garis nol
                fig_residual.add_hline(y=0, line_dash='dash', line_color='gray')

                # Update layout
                fig_residual.update_layout(
                    title=None,
                    xaxis_title='Tahun',
                    yaxis_title='Residual (ton)',
                    template='plotly_white',
                    barmode='group',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=400
                )

                st.plotly_chart(fig_residual, use_container_width=True)
                
            with tab2:
                # Tampilkan tabel residual dengan format yang lebih baik
                st.dataframe(
                    residual_df.style.format({
                        'Catch Aktual (ton)': '{:.2f}',
                        'Schaefer Pred. (ton)': '{:.2f}',
                        'Fox Pred. (ton)': '{:.2f}',
                        'Residual Schaefer': '{:.2f}',
                        'Residual Fox': '{:.2f}'
                    }).background_gradient(
                        subset=['Residual Schaefer', 'Residual Fox'], 
                        cmap='RdYlGn', 
                        vmin=-max(abs(residual_df['Residual Schaefer'].max()), abs(residual_df['Residual Schaefer'].min())),
                        vmax=max(abs(residual_df['Residual Schaefer'].max()), abs(residual_df['Residual Schaefer'].min()))
                    ),
                    use_container_width=True
                )

            # Kesimpulan
            st.subheader("🎯 Kesimpulan Evaluasi Model", anchor=False)

            # Tentukan model terbaik berdasarkan RMSE
            best_model = 'Schaefer' if schaefer_metrics['RMSE'] < fox_metrics['RMSE'] else 'Fox'
            best_r2 = schaefer_results['R2'] if best_model == 'Schaefer' else fox_results['R2']
            best_rmse = schaefer_metrics['RMSE'] if best_model == 'Schaefer' else fox_metrics['RMSE']
            best_msy = schaefer_results['CMSY'] if best_model == 'Schaefer' else fox_results['CMSY']
            best_eopt = schaefer_results['Eopt'] if best_model == 'Schaefer' else fox_results['Eopt']

            # Tampilkan hasil dengan cards
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Model Terbaik: {best_model}**\n- RMSE: {best_rmse:.3f} ton\n- R²: {best_r2:.4f}")

            with col2:
                st.success(f"**Rekomendasi Pengelolaan:**\n- MSY: {best_msy:.2f} ton\n- E-opt: {best_eopt:.2f} hari")

            # Tambahkan interpretasi status pemanfaatan sumber daya
            recent_years = yearly_data.sort_values('tahun', ascending=False).head(3)
            recent_avg_effort = recent_years['effort (hari)'].mean()
            effort_ratio = recent_avg_effort / best_eopt

            # Tentukan status dan warna yang sesuai
            if effort_ratio < 0.5:
                status = "Under-exploited (Pemanfaatan rendah)"
                color = "green"
            elif effort_ratio < 0.9:
                status = "Moderately exploited (Pemanfaatan sedang)"
                color = "blue"
            elif effort_ratio < 1.1:
                status = "Fully exploited (Pemanfaatan penuh)"
                color = "orange"
            elif effort_ratio < 1.5:
                status = "Over-exploited (Pemanfaatan berlebih)"
                color = "red"
            else:
                status = "Severely over-exploited (Pemanfaatan sangat berlebih)"
                color = "darkred"

            # Visualisasi gauge untuk status pemanfaatan
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = effort_ratio,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Status Pemanfaatan Sumber Daya", 'font': {'size': 18}},
                delta = {'reference': 1, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.5], 'color': 'rgba(0, 128, 0, 0.3)'},
                        {'range': [0.5, 0.9], 'color': 'rgba(30, 144, 255, 0.3)'},
                        {'range': [0.9, 1.1], 'color': 'rgba(255, 165, 0, 0.3)'},
                        {'range': [1.1, 1.5], 'color': 'rgba(255, 0, 0, 0.3)'},
                        {'range': [1.5, 2], 'color': 'rgba(128, 0, 0, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 1
                    }
                }
            ))

            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Pindahkan "Detail Status Pemanfaatan" keluar dari expander untuk menghindari nested expander
        st.subheader("📝 Detail Status Pemanfaatan")
        st.write(f"""
        - Rata-rata effort 3 tahun terakhir: **{recent_avg_effort:.2f}** hari
        - Effort optimal (E-opt): **{best_eopt:.2f}** hari
        - Rasio terhadap E-opt: **{effort_ratio:.2f}**
        - Status: **{status}**

        **Rekomendasi Tindakan:**
        """)

        if effort_ratio < 0.5:
            st.success("Peningkatan upaya penangkapan masih dapat dilakukan untuk mencapai pemanfaatan optimal.")
        elif effort_ratio < 0.9:
            st.info("Peningkatan upaya penangkapan dapat dilakukan dengan hati-hati, dengan pemantauan yang ketat terhadap dampak peningkatan.")
        elif effort_ratio < 1.1:
            st.warning("Upaya penangkapan sudah optimal. Pertahankan tingkat upaya saat ini dan lakukan pemantauan rutin.")
        elif effort_ratio < 1.5:
            st.error("Pengurangan upaya penangkapan diperlukan untuk mencapai pemanfaatan yang berkelanjutan.")
        else:
            st.error("Pengurangan upaya penangkapan yang signifikan sangat diperlukan untuk mencegah kerusakan stok lebih lanjut.")
                    
    else:
        st.info('Please upload a CSV file to proceed.')
        



elif menu == 'About':
    # Header with custom styling
    st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight:bold;
        color:#1E88E5;
        margin-bottom:0px;
    }
    .sub-font {
        font-size:20px;
        color:#424242;
        margin-top:0px;
    }
    .section-header {
        background-color:#f0f8ff;
        padding:10px;
        border-radius:5px;
        margin-top:30px;
        border-left:5px solid #1E88E5;
    }
    .card {
        background-color:white;
        padding:20px;
        border-radius:10px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-bottom:20px;
        border-top:4px solid #1E88E5;
    }
    .model-card {
        background-color:#f8f9fa;
        padding:15px;
        border-radius:8px;
        margin-bottom:15px;
    }
    .feature-item {
        margin-bottom:10px;
    }
    </style>
    
    <p class="big-font">SISTOK</p>
    <p class="sub-font">Sistem Informasi Stok Perikanan</p>
    """, unsafe_allow_html=True)
    
    # Main Description with card styling
    st.markdown("""
    <div class="card">
    <h3>What is SISTOK?</h3>
    <p>SISTOK is a comprehensive web-based application designed to help fisheries researchers, managers, and stakeholders analyze and understand fish stock data effectively. This tool provides various analytical capabilities to support sustainable fisheries management decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features with better visual organization
    st.markdown('<div class="section-header"><h2>✨ Key Features</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>📊 Data Management</h3>
        <div class="feature-item">• <b>CSV file upload</b> with intelligent parsing</div>
        <div class="feature-item">• <b>Real-time data processing</b> for immediate insights</div>
        <div class="feature-item">• <b>Interactive data filtering</b> capabilities</div>
        </div>
        
        <div class="card">
        <h3>📈 Visualization</h3>
        <div class="feature-item">• <b>Dynamic charts and graphs</b> for data exploration</div>
        <div class="feature-item">• <b>Catch statistics visualization</b> with multiple views</div>
        <div class="feature-item">• <b>Temporal trend analysis</b> for pattern identification</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🎯 Advanced Analytics</h3>
        <div class="feature-item">• <b>Surplus Production Models</b> (Schaefer & Fox)</div>
        <div class="feature-item">• <b>CPUE Analysis</b> with standardization options</div>
        <div class="feature-item">• <b>Fishing effort standardization</b> across gear types</div>
        </div>
        
        <div class="card">
        <h3>📆 Time Series Analysis</h3>
        <div class="feature-item">• <b>Multiple time frame options</b> for different perspectives</div>
        <div class="feature-item">• <b>Trend identification</b> using advanced algorithms</div>
        <div class="feature-item">• <b>Seasonal pattern analysis</b> for temporal insights</div>
        </div>
        """, unsafe_allow_html=True)

    # How to use - redesigned with modern expander
    st.markdown('<div class="section-header"><h2>🔍 How to Use SISTOK</h2></div>', unsafe_allow_html=True)

    with st.expander("**STEP-BY-STEP GUIDE**"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
            <h3>1. Data Upload</h3>
            <div class="feature-item">• Navigate to the <b>Analysis</b> section</div>
            <div class="feature-item">• Upload your CSV file containing fishing data</div>
            <div class="feature-item">• Ensure your data includes required columns (date, catch, effort)</div>
            </div>
            
            <div class="card">
            <h3>2. Data Exploration</h3>
            <div class="feature-item">• Use the <b>Dashboard</b> to view data overview</div>
            <div class="feature-item">• Apply filters to focus on specific time periods, ports, or fish species</div>
            <div class="feature-item">• Examine catch trends and patterns through visualizations</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
            <h3>3. Analysis</h3>
            <div class="feature-item">• Calculate CPUE for different fishing gears</div>
            <div class="feature-item">• Apply surplus production models for stock assessment</div>
            <div class="feature-item">• Estimate MSY and optimal fishing effort levels</div>
            </div>
            
            <div class="card">
            <h3>4. Results Interpretation</h3>
            <div class="feature-item">• Review visualizations and statistical metrics</div>
            <div class="feature-item">• Reliability Model Evaluation</div>
            <div class="feature-item">• Make informed fisheries management decisions</div>
            </div>
            """, unsafe_allow_html=True)

    # Access Steps - newly designed section
    st.markdown('<div class="section-header"><h2>🚪 Access Steps</h2></div>', unsafe_allow_html=True)
    
    with st.expander("**HOW TO ACCESS SISTOK**"):
        st.markdown("""
        <div class="card">
        <h3>1. Access the Web Application</h3>
        <div class="feature-item">• Open your web browser and navigate to the SISTOK application URL</div>
        <div class="feature-item">• For local access: <code>https://sistok-tools-v1.streamlit.app/</code></div>
        <div class="feature-item">• For remote access: [Application URL]</div>
        </div>
        

        
        <div class="card">
        <h3>2. Navigation</h3>
        <div class="feature-item">• Use the sidebar menu to navigate between different modules</div>
        <div class="feature-item">• Select <b>Dashboard</b> for overview statistics and trends</div>
        <div class="feature-item">• Select <b>Analysis</b> for detailed data processing workflows</div>
        <div class="feature-item">• Select <b>About</b> for application description and others</div>
        </div>
        """, unsafe_allow_html=True)

    # Graphical Representation/Models - redesigned with tabs
    st.markdown('<div class="section-header"><h2>📊 Model Representations</h2></div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Schaefer Model", "Fox Model"])
    
    with tab1:
        st.markdown("""
        <div class="model-card">
        <h3 style="color:#1E88E5;">Schaefer Model</h3>
        <p>The Schaefer model is based on the logistic growth equation and assumes a linear relationship between CPUE (Catch Per Unit Effort) and effort (E).</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Mathematical Formulation")
            st.latex(r'''CPUE = a - b \times E''')
            st.latex(r'''Y = a \times E - b \times E^2''')
            st.latex(r'''MSY = \frac{a^2}{4b}''')
            st.latex(r'''E_{MSY} = \frac{a}{2b}''')
            
        with col2:
            st.markdown("#### Model Characteristics")
            st.markdown("""
            <div class="feature-item">• CPUE decreases <b>linearly</b> with increasing fishing effort</div>
            <div class="feature-item">• Yield (Y) follows a <b>parabolic relationship</b> with effort</div>
            <div class="feature-item">• MSY occurs at <b>half the effort level</b> that would drive the stock to zero</div>
            <div class="feature-item">• <b>More conservative</b> than the Fox model at higher effort levels</div>
            """, unsafe_allow_html=True)
        
    with tab2:
        st.markdown("""
        <div class="model-card">
        <h3 style="color:#1E88E5;">Fox Model</h3>
        <p>The Fox model assumes an exponential relationship between CPUE and effort, allowing for more sustainable harvesting at lower population levels.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Mathematical Formulation")
            st.latex(r'''CPUE = a \times e^{-b \times E}''')
            st.latex(r'''Y = a \times E \times e^{-b \times E}''')
            st.latex(r'''MSY = \frac{a}{b \times e} \times e^{-1}''')
            st.latex(r'''E_{MSY} = \frac{1}{b}''')
            
        with col2:
            st.markdown("#### Model Characteristics")
            st.markdown("""
            <div class="feature-item">• CPUE decreases <b>exponentially</b> with increasing fishing effort</div>
            <div class="feature-item">• Better suited for stocks that can sustain fishing pressure at <b>low population levels</b></div>
            <div class="feature-item">• MSY occurs at a <b>higher effort level</b> compared to the Schaefer model</div>
            <div class="feature-item">• Often provides a <b>better fit</b> for certain fisheries data</div>
            """, unsafe_allow_html=True)

    # Terminology - redesigned with modern styling
    st.markdown('<div class="section-header"><h2>📚 Key Terminology</h2></div>', unsafe_allow_html=True)
    
    with st.expander("**FISHERIES MANAGEMENT TERMS**"):
        col1, col2 = st.columns(2)
        
        terms = {
            "CPUE (Catch Per Unit Effort)": "A measure of the abundance of a target species, calculated as the total catch divided by the total fishing effort.",
            "MSY (Maximum Sustainable Yield)": "The largest yield (or catch) that can be taken from a species' stock over an indefinite period without depleting the stock.",
            "Fishing Effort": "The amount of fishing gear of a specific type used on the fishing grounds over a given unit of time.",
            "Biomass": "The total weight of fish in a stock or population.",
            "Carrying Capacity (K)": "The maximum population size of the species that the environment can sustain indefinitely.",
            "Growth Rate (r)": "The intrinsic rate of population increase in the absence of density-dependent factors.",
            "Stock Assessment": "The process of collecting and analyzing biological and statistical information to determine the changes in the abundance of fishery stocks in response to fishing.",
            "Overfishing": "Fishing activity that leads to a reduction in stock levels below acceptable levels.",
            "Surplus Production": "The total weight (biomass) produced by a fish population through growth and reproduction in excess of what is needed to maintain the population size."
        }
        
        # Split terms into two columns
        terms_list = list(terms.items())
        mid_point = len(terms_list) // 2
        
        with col1:
            for term, definition in terms_list[:mid_point]:
                st.markdown(f"""
                <div class="card">
                <h4>{term}</h4>
                <p>{definition}</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            for term, definition in terms_list[mid_point:]:
                st.markdown(f"""
                <div class="card">
                <h4>{term}</h4>
                <p>{definition}</p>
                </div>
                """, unsafe_allow_html=True)

    # Model Evaluation - redesigned with tabs for each metric
    st.markdown('<div class="section-header"><h2>📏 Model Evaluation Metrics</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
    <p>SISTOK uses multiple statistical metrics to evaluate and compare the performance of fisheries models. 
    Each metric provides unique insights into how well a model fits the observed data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    metric_tabs = st.tabs(["RMSE", "MAE", "MAPE", "R²"])
    
    with metric_tabs[0]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Root Mean Square Error")
            st.latex(r'''RMSE = \sqrt{\frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{n}}''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>RMSE measures the average magnitude of the errors in predictions, with a higher penalty for large errors.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Lower values indicate better model fit</div>
            <div class="feature-item">• Expressed in the same units as the dependent variable</div>
            <div class="feature-item">• Gives higher weight to larger errors due to squaring</div>
            <div class="feature-item">• Useful when large errors are particularly undesirable</div>
            </div>
            """, unsafe_allow_html=True)
    
    with metric_tabs[1]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Mean Absolute Error")
            st.latex(r'''MAE = \frac{\sum_{i=1}^{n} |y_i - \hat{y}_i|}{n}''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>MAE measures the average magnitude of errors in predictions without considering their direction.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Lower values indicate better model fit</div>
            <div class="feature-item">• Expressed in the same units as the dependent variable</div>
            <div class="feature-item">• Treats all errors equally (no squaring)</div>
            <div class="feature-item">• More robust to outliers than RMSE</div>
            </div>
            """, unsafe_allow_html=True)
            
    with metric_tabs[2]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Mean Absolute Percentage Error")
            st.latex(r'''MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>MAPE expresses prediction accuracy as a percentage of error, making it easy to interpret across different scales.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Lower values indicate better model fit</div>
            <div class="feature-item">• Provides error in percentage terms (scale-independent)</div>
            <div class="feature-item">• Easier to communicate to non-technical stakeholders</div>
            <div class="feature-item">• Not suitable when actual values are close to zero</div>
            </div>
            """, unsafe_allow_html=True)
            
    with metric_tabs[3]:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("### Coefficient of Determination")
            st.latex(r'''R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}''')
        
        with col2:
            st.markdown("""
            <div class="card">
            <h4>What it measures</h4>
            <p>R² indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.</p>
            
            <h4>Interpretation</h4>
            <div class="feature-item">• Values range from 0 to 1 (or 0% to 100%)</div>
            <div class="feature-item">• Higher values indicate better model fit</div>
            <div class="feature-item">• R² = 1: model explains all variability in the response data</div>
            <div class="feature-item">• R² = 0: model explains none of the variability</div>
            </div>
            """, unsafe_allow_html=True)

    # Technical Requirements - redesigned with icon and card styling
    st.markdown('<div class="section-header"><h2>💻 Technical Requirements</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h3>Data Format Specifications</h3>
    <p>Your data should be in CSV format with the following columns:</p>
    <div class="feature-item">• <b>Date columns</b>: tanggal_berangkat, tanggal_kedatangan</div>
    <div class="feature-item">• <b>Catch data</b>: berat</div>
    <div class="feature-item">• <b>Effort data</b>: Jumlah hari</div>
    <div class="feature-item">• <b>Fishing gear</b>: jenis_api</div>
    <div class="feature-item">• Additional metadata as needed</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Contact Information - redesigned with better styling
    st.markdown('<div class="section-header"><h2>📬 Contact & Support</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>Developer Contact</h3>
        <div class="feature-item">📧 <b>Email</b>: tandrysimamora@gmail.com</div>
        <div class="feature-item">📱 <b>Phone</b>: +62 822 6160-6428</div>
        <div class="feature-item">🌐 <b>Website</b>: https://tndry.github.io/portfolio-website/</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:  
        st.markdown("""
        <div class="card">
        <h3>Support Resources</h3>
        <div class="feature-item">📚 <b>Documentation</b>: [Link to documentation]</div>
        <div class="feature-item">❓ <b>FAQ</b>: [Link to FAQ page]</div>
        <div class="feature-item">🎓 <b>Tutorials</b>: [Link to tutorials]</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Version info - moved to the main content area with better styling
    st.markdown("""
    <div style="margin-top:30px; text-align:center; padding:10px; background-color:#f0f8ff; border-radius:5px;">
    <p style="margin:0; color:#1E88E5; font-weight:bold;">SISTOK v2.0.0</p>
    <p style="margin:0; font-size:12px; color:#616161;">Last updated: April 2025</p>
    </div>
    """, unsafe_allow_html=True)
        

else:
    st.error('Data tidak tersedia. Silahkan periksa kembali file Anda.')




    