from pathlib import Path
from datetime import datetime, time, timedelta
import json
from urllib.parse import urlencode
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'traffic_model.joblib'
METADATA_PATH = BASE_DIR / 'model_metadata.json'
DATA_PATH = BASE_DIR / 'traffic_data.csv'

st.set_page_config(page_title='Lagos Traffic Intelligence', page_icon='🚦', layout='wide', initial_sidebar_state='expanded')

st.markdown('''
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
.hero {padding: 1.4rem 1.6rem; border-radius: 20px; background: linear-gradient(120deg,#071a2d,#123b59); color:white; margin-bottom:1.1rem; box-shadow:0 8px 28px rgba(0,0,0,.12)}
.hero h1 {margin:0; font-size:2rem}.hero p {margin:.35rem 0 0; opacity:.82}
.card {padding:1.1rem 1.2rem; border:1px solid rgba(128,128,128,.18); border-radius:18px; background:rgba(128,128,128,.035); margin-bottom:.8rem}
.result-low,.result-moderate,.result-high,.result-severe {padding:1.2rem 1.4rem;border-radius:18px;color:white;margin:1rem 0}
.result-low{background:linear-gradient(120deg,#087f5b,#20c997)} .result-moderate{background:linear-gradient(120deg,#b7791f,#f59f00)}
.result-high{background:linear-gradient(120deg,#c2410c,#f76707)} .result-severe{background:linear-gradient(120deg,#991b1b,#e03131)}
.smallmuted {opacity:.72;font-size:.9rem}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.16);padding:1rem;border-radius:16px;background:rgba(128,128,128,.025)}
.stButton>button, .stFormSubmitButton>button {border-radius:12px; font-weight:700;}
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text(encoding='utf-8'))

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()
    df['level'] = pd.cut(df['congestion_ratio'], [-np.inf,1.15,1.40,1.80,np.inf], labels=['Low','Moderate','High','Severe'], right=False)
    return df

def congestion_level(ratio):
    if ratio < 1.15: return 'Low', 'Traffic should move close to free-flow conditions.'
    if ratio < 1.40: return 'Moderate', 'Some delay is expected, but movement should remain manageable.'
    if ratio < 1.80: return 'High', 'Significant congestion and slower travel are expected.'
    return 'Severe', 'Very heavy congestion is expected. Consider changing departure time if possible.'

def make_features(route, weather, selected_date, selected_time):
    dt = datetime.combine(selected_date, selected_time)
    mins = dt.hour*60 + dt.minute
    return pd.DataFrame([{'route':route,'weather':weather,'day':dt.strftime('%A'),'hour':dt.hour,'minute':dt.minute,
        'month':dt.month,'day_of_month':dt.day,'is_weekend':int(dt.weekday()>=5),
        'time_sin':np.sin(2*np.pi*mins/1440),'time_cos':np.cos(2*np.pi*mins/1440)}])


def find_next_easing(model, metadata, route, weather, selected_date, selected_time,
                     current_pred, step_minutes=15, max_hours=6):
    """Find the nearest later time with meaningfully easier predicted traffic."""
    start_dt = datetime.combine(selected_date, selected_time)
    current_level, _ = congestion_level(current_pred)
    if current_level not in ("High", "Severe"):
        return None

    target = 1.80 if current_level == "Severe" else 1.40
    best = None

    for minutes_ahead in range(step_minutes, max_hours * 60 + step_minutes, step_minutes):
        candidate_dt = start_dt + timedelta(minutes=minutes_ahead)
        features = make_features(route, weather, candidate_dt.date(), candidate_dt.time())
        ratio = float(model.predict(features)[0])
        ratio = float(np.clip(ratio, metadata["target_min"], metadata["target_max"]))

        if best is None or ratio < best["ratio"]:
            best = {"datetime": candidate_dt, "ratio": ratio, "wait_minutes": minutes_ahead}

        if ratio < target and ratio <= current_pred - 0.10:
            level, _ = congestion_level(ratio)
            return {"datetime": candidate_dt, "ratio": ratio,
                    "level": level, "wait_minutes": minutes_ahead}

    if best and best["ratio"] <= current_pred - 0.10:
        best["level"] = congestion_level(best["ratio"])[0]
        return best
    return None


def format_wait(minutes):
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours} hr {mins} min"
    if hours:
        return f"{hours} hr"
    return f"{mins} min"


def make_google_calendar_url(reminder_dt, easing_dt, route):
    """Build a pre-filled Google Calendar event URL."""
    start = reminder_dt.strftime("%Y%m%dT%H%M%S")
    end = (reminder_dt + timedelta(minutes=10)).strftime("%Y%m%dT%H%M%S")
    details = (
        f"Traffic on {route} was predicted to ease around "
        f"{easing_dt.strftime('%I:%M %p')}. Check the traffic prediction before leaving."
    )
    params = {
        "action": "TEMPLATE",
        "text": "Check traffic before leaving",
        "dates": f"{start}/{end}",
        "details": details,
        "ctz": "Africa/Lagos",
    }
    return "https://calendar.google.com/calendar/render?" + urlencode(params)


def make_ics_reminder(reminder_dt, easing_dt, route):
    """Create a calendar reminder file usable on phones and computers."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    start = reminder_dt.strftime("%Y%m%dT%H%M%S")
    end = (reminder_dt + timedelta(minutes=10)).strftime("%Y%m%dT%H%M%S")
    description = (
        f"Traffic on {route} was predicted to ease around "
        f"{easing_dt.strftime('%I:%M %p')}."
    )
    content = (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//Lagos Traffic Intelligence//EN\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{stamp}-traffic-easing@lagostraffic\r\n"
        f"DTSTAMP:{stamp}\r\n"
        f"DTSTART:{start}\r\n"
        f"DTEND:{end}\r\n"
        "SUMMARY:Traffic easing reminder\r\n"
        f"DESCRIPTION:{description}\r\n"
        "BEGIN:VALARM\r\n"
        "TRIGGER:-PT0M\r\n"
        "ACTION:DISPLAY\r\n"
        "DESCRIPTION:Check traffic before leaving\r\n"
        "END:VALARM\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    return content.encode("utf-8")


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def base_features(df):
    x = pd.DataFrame()
    x['route']=df['route']; x['weather']=df['weather']; x['day']=df['timestamp'].dt.day_name()
    x['hour']=df['timestamp'].dt.hour; x['minute']=df['timestamp'].dt.minute; x['month']=df['timestamp'].dt.month
    x['day_of_month']=df['timestamp'].dt.day; x['is_weekend']=(df['timestamp'].dt.dayofweek>=5).astype(int)
    mins=x['hour']*60+x['minute']; x['time_sin']=np.sin(2*np.pi*mins/1440); x['time_cos']=np.cos(2*np.pi*mins/1440)
    return x

try:
    model, metadata, df = load_model(), load_metadata(), load_data()
except Exception as e:
    st.error(f'Could not load the project files: {e}'); st.stop()

with st.sidebar:
    st.title('🚦 Lagos Traffic')
    st.caption('Traffic Intelligence System')
    page = st.radio('Navigation', ['Dashboard','Predict Traffic','Analytics','Historical Data','Model Performance','About'], label_visibility='collapsed')
    st.divider()
    st.caption('ROUTE')
    st.write('**Berger → Lagos Island**')
    st.caption(f"Data: {pd.to_datetime(metadata['data_start']).strftime('%d %b')} – {pd.to_datetime(metadata['data_end']).strftime('%d %b %Y')}")
    st.caption(f"{len(df):,} traffic observations")

if page == 'Dashboard':
    hero('Traffic Intelligence Dashboard','Berger → Lagos Island • Historical traffic and weather analytics powered by machine learning')
    avg=df.congestion_ratio.mean(); peak=df.groupby('hour').congestion_ratio.mean().idxmax(); maxr=df.congestion_ratio.max()
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Traffic records',f'{len(df):,}'); c2.metric('Average congestion',f'{avg:.2f}'); c3.metric('Peak hour',f'{int(peak):02d}:00'); c4.metric('Highest ratio',f'{maxr:.2f}')
    st.subheader('Traffic overview')
    daily=df.set_index('timestamp').resample('D').congestion_ratio.mean().reset_index()
    fig=px.line(daily,x='timestamp',y='congestion_ratio',markers=True,labels={'timestamp':'Date','congestion_ratio':'Average congestion ratio'})
    fig.update_layout(height=380,margin=dict(l=10,r=10,t=20,b=10)); st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2)
    hourly=df.groupby('hour',as_index=False).congestion_ratio.mean()
    fig=px.bar(hourly,x='hour',y='congestion_ratio',labels={'hour':'Hour of day','congestion_ratio':'Average congestion ratio'}); fig.update_layout(height=350,margin=dict(l=10,r=10,t=20,b=10)); a.plotly_chart(fig,use_container_width=True)
    weather=df.groupby('weather',as_index=False).congestion_ratio.mean().sort_values('congestion_ratio',ascending=False)
    fig=px.bar(weather,x='congestion_ratio',y='weather',orientation='h',labels={'weather':'Weather','congestion_ratio':'Average congestion ratio'}); fig.update_layout(height=350,margin=dict(l=10,r=10,t=20,b=10)); b.plotly_chart(fig,use_container_width=True)

elif page == 'Predict Traffic':
    hero('Traffic Congestion Prediction','Choose a date, time and weather condition to estimate congestion on Berger → Lagos Island.')
    left,right=st.columns([1,1.15],gap='large')
    with left:
        st.subheader('Trip details')
        with st.form('prediction'):
            route=st.selectbox('Route',metadata['routes']); weather=st.selectbox('Weather condition',metadata['weather_conditions'])
            d=st.date_input('Travel date',datetime.now().date()); t=st.time_input('Travel time',time(8,0),step=900)
            submitted=st.form_submit_button('Predict congestion',use_container_width=True)
    with right:
        st.subheader('Prediction result')
        if submitted:
            pred=float(model.predict(make_features(route,weather,d,t))[0]); pred=float(np.clip(pred,metadata['target_min'],metadata['target_max']))
            level,msg=congestion_level(pred); pct=max(0,(pred-1)*100); css=level.lower()
            st.markdown(f'<div class="result-{css}"><div class="smallmuted">PREDICTED TRAFFIC</div><h2 style="margin:.2rem 0">{level.upper()} CONGESTION</h2><div style="font-size:2.4rem;font-weight:800">{pred:.2f}</div><div>Congestion ratio • approximately {pct:.0f}% above free-flow travel time</div></div>',unsafe_allow_html=True)
            st.info(msg)
            period='Morning peak' if 6<=t.hour<10 else 'Evening peak' if 16<=t.hour<21 else 'Off-peak / other period'
            c1,c2=st.columns(2); c1.metric('Weather',weather); c2.metric('Travel period',period)
            st.caption(f"{d.strftime('%A, %d %B %Y')} at {t.strftime('%H:%M')} • {route}")

            easing = find_next_easing(model, metadata, route, weather, d, t, pred)
            if level in ('High', 'Severe'):
                st.markdown('Smart departure suggestion')
                if easing:
                    easing_dt = easing['datetime']
                    wait_text = format_wait(easing['wait_minutes'])
                    st.success(
                        f"Traffic is predicted to ease to **{easing['level']}** around "
                        f"**{easing_dt.strftime('%I:%M %p')}** "
                        f"(predicted ratio **{easing['ratio']:.2f}**). "
                        f"That is approximately **{wait_text}** after your selected departure time."
                    )
                    st.caption(
                        "This is a model-based estimate, not a guarantee. Weather and live road "
                        "conditions can change after the prediction."
                    )

                    reminder_dt = easing_dt - timedelta(minutes=15)
                    if reminder_dt <= datetime.combine(d, t):
                        reminder_dt = easing_dt

                    google_calendar_url = make_google_calendar_url(
                        reminder_dt, easing_dt, route
                    )
                    ics = make_ics_reminder(reminder_dt, easing_dt, route)

                    st.markdown("#### Set a traffic reminder")
                    st.caption(
                        f"The reminder is set for {reminder_dt.strftime('%I:%M %p')}, "
                        f"15 minutes before the predicted easing time of {easing_dt.strftime('%I:%M %p')}."
                    )

                    cal1, cal2 = st.columns(2)
                    with cal1:
                        st.link_button(
                            "Add to Google Calendar",
                            google_calendar_url,
                            use_container_width=True,
                            help="Opens Google Calendar with the reminder details already filled in."
                        )
                    with cal2:
                        st.download_button(
                            "Download calendar reminder",
                            data=ics,
                            file_name="traffic_easing_reminder.ics",
                            mime="text/calendar",
                            use_container_width=True,
                            help="Use this if Google Calendar is unavailable. Open the downloaded file with your calendar app."
                        )

                    st.caption(
                        "Google Calendar is the quickest option"
                        
                    )
                else:
                    st.warning(
                        "The model did not find a clearly better traffic period within the next "
                        "6 hours under the selected weather condition."
                    )
        else:
            st.markdown('<div class="card"><h3>Ready to predict</h3><p>Enter the trip details and select <b>Predict congestion</b>. Your result and travel guidance will appear here.</p></div>',unsafe_allow_html=True)

elif page == 'Analytics':
    hero('Traffic Analytics','Explore patterns by hour, day, weather and congestion level.')
    tab1,tab2,tab3,tab4=st.tabs(['⏱ Hourly','📅 Day of week','🌧 Weather','🚥 Congestion levels'])
    with tab1:
        g=df.groupby('hour',as_index=False).congestion_ratio.mean(); fig=px.line(g,x='hour',y='congestion_ratio',markers=True,labels={'hour':'Hour of day','congestion_ratio':'Average congestion ratio'}); fig.update_layout(height=470); st.plotly_chart(fig,use_container_width=True)
    with tab2:
        order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']; g=df.groupby('day',as_index=False).congestion_ratio.mean(); g['day']=pd.Categorical(g.day,categories=order,ordered=True); g=g.sort_values('day'); fig=px.bar(g,x='day',y='congestion_ratio',labels={'day':'Day','congestion_ratio':'Average congestion ratio'}); fig.update_layout(height=470); st.plotly_chart(fig,use_container_width=True)
    with tab3:
        g=df.groupby('weather',as_index=False).agg(avg_congestion=('congestion_ratio','mean'),avg_delay=('traffic_delay_s','mean'),records=('congestion_ratio','size')).sort_values('avg_congestion',ascending=False)
        fig=px.bar(g,x='weather',y='avg_congestion',hover_data=['avg_delay','records'],labels={'weather':'Weather condition','avg_congestion':'Average congestion ratio'}); fig.update_layout(height=470); st.plotly_chart(fig,use_container_width=True); st.dataframe(g.rename(columns={'weather':'Weather','avg_congestion':'Avg congestion','avg_delay':'Avg delay (s)','records':'Records'}),use_container_width=True,hide_index=True)
    with tab4:
        counts=df.level.value_counts().reindex(['Low','Moderate','High','Severe']).fillna(0).reset_index(); counts.columns=['Level','Records']; fig=px.pie(counts,names='Level',values='Records',hole=.55); fig.update_layout(height=470); st.plotly_chart(fig,use_container_width=True)

elif page == 'Historical Data':
    hero('Historical Traffic Data','Filter and inspect the observations used for traffic analysis.')
    c1,c2,c3=st.columns(3)
    days=c1.multiselect('Day',sorted(df.day.unique())); weathers=c2.multiselect('Weather',sorted(df.weather.unique())); levels=c3.multiselect('Congestion level',['Low','Moderate','High','Severe'])
    view=df.copy()
    if days:view=view[view.day.isin(days)]
    if weathers:view=view[view.weather.isin(weathers)]
    if levels:view=view[view.level.astype(str).isin(levels)]
    st.write(f'**{len(view):,} records shown**')
    cols=['timestamp','day','weather','travel_time_s','traffic_delay_s','traffic_length_m','congestion_ratio','level']
    st.dataframe(view[cols],use_container_width=True,hide_index=True,height=520)
    st.download_button('Download filtered CSV',view[cols].to_csv(index=False).encode(),file_name='filtered_traffic_data.csv',mime='text/csv')

elif page == 'Model Performance':
    hero('Model Performance','Understand how the Random Forest model performs on unseen chronological test data.')
    m=metadata['metrics']; c1,c2,c3,c4=st.columns(4); c1.metric('R²',f"{m['R2']:.3f}"); c2.metric('MAE',f"{m['MAE']:.3f}"); c3.metric('RMSE',f"{m['RMSE']:.3f}"); c4.metric('Test records',f"{metadata['test_rows']:,}")
    split=int(len(df)*.8); test=df.sort_values('timestamp').iloc[split:].copy(); test['Predicted']=model.predict(base_features(test)); test['Actual']=test.congestion_ratio
    a,b=st.columns(2)
    fig=px.scatter(test,x='Actual',y='Predicted',opacity=.6); mn=min(test.Actual.min(),test.Predicted.min()); mx=max(test.Actual.max(),test.Predicted.max()); fig.add_trace(go.Scatter(x=[mn,mx],y=[mn,mx],mode='lines',name='Perfect prediction')); fig.update_layout(height=420); a.plotly_chart(fig,use_container_width=True)
    timeline=test[['timestamp','Actual','Predicted']].set_index('timestamp').resample('6h').mean().reset_index(); fig=px.line(timeline,x='timestamp',y=['Actual','Predicted'],labels={'value':'Congestion ratio','timestamp':'Time','variable':'Series'}); fig.update_layout(height=420); b.plotly_chart(fig,use_container_width=True)
    st.info('The model uses route, weather, day/date and time features. Travel time and traffic delay are intentionally excluded from the prediction inputs to avoid data leakage.')

else:
    hero('About the Project','Machine-learning traffic congestion prediction for the Berger → Lagos Island corridor.')
    st.markdown('''### Project purpose
This application uses historical traffic observations and recorded weather conditions to predict the likely congestion ratio for a selected date and time on the **Berger → Lagos Island** route.

### How it works
The prediction model is a **Random Forest Regressor**. Inputs include the route, weather condition, day of the week, hour, minute, month, day of month, weekend indicator and cyclical time features. The dataset is split chronologically: the first 80% is used for training and the final 20% for testing.

### Important limitation
The current dataset contains only the Berger → Lagos Island route. Predictions should therefore be treated as route-specific. Adding other routes requires collecting data for those routes and retraining the model.
''')
    c1,c2,c3=st.columns(3); c1.metric('Training records',f"{metadata['training_rows']:,}"); c2.metric('Test records',f"{metadata['test_rows']:,}"); c3.metric('Weather categories',len(metadata['weather_conditions']))
