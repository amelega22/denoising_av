import requests
from obspy import UTCDateTime
from obspy import Stream
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv
import pandas as pd
from libcomcat.search import get_event_by_id
from libcomcat.dataframes import get_phase_dataframe
from obspy.signal.trigger import classic_sta_lta
rcParams.update({'font.size': 10})

# Query parameters
URL = 'https://earthquake.usgs.gov/fdsnws/event/1/query'
starttime = UTCDateTime(2022, 11, 18) #yellow on 11/18
endtime = UTCDateTime(2022, 11, 19) #back to green on 2023/7/18
volcano_latitude =  51.869504
volcano_longitude = -178.079509
contributor = 'av'
SOURCE = 'EarthScope'
NETWORK = 'AV'
STATION = 'TASE,TANO'#
LOCATION = '*'
CHANNEL = 'BHZ'
SVDIR = '/Users/dfee/Documents/AVO/denoising/figs/'

# set earthquake parameters
radius_km = 25
radius_min = 0
maxdepth = 40
MAG_MIN = 2 # start with 1?

FILT = 0.1
PRE_WIN = 10
DUR = 60
PAD = 10    # pad waveform for tapering/etc

STA = 1
LTA = 10
#STALTA_THRESH = 4

PICK_QUERY = True
SAVE = True

#%% read in catalog data

print('Searching COMCAT...')

session = requests.Session()
catalog_params = {
    'format': 'csv',
    'starttime': starttime.strftime('%Y-%m-%dT%H:%M:%S'),
    'endtime': endtime.strftime('%Y-%m-%dT%H:%M:%S'),
    'latitude': volcano_latitude,
    'longitude': volcano_longitude,
    'maxradiuskm': radius_km,
    'minradiuskm': radius_min,
    'maxdepth': maxdepth,
    'minmagnitude': MAG_MIN,
    'eventtype': 'earthquake',
    'reviewstatus': 'reviewed',
    'contributor': contributor,
    'catalog': contributor
}

catalog_request = session.get(URL, params=catalog_params, timeout=60)
catalog_request.raise_for_status()
csv_dict = csv.DictReader(catalog_request.iter_lines(decode_unicode=True))
events = list(csv_dict)

# Store event ids, types, and statuses
evids = []
types = []
status = []
count = 0
eq_times = []
eq_mags = []
eq_depths = []
eq_lats = []
eq_lons = []

for event_dict in events:
    # only choose reviewed events
    if event_dict['status'] == 'reviewed':
        count += 1
        evids.append(event_dict['id'])
        types.append(event_dict['type'])
        status.append(event_dict['status'])

        print('Event ID: %s, Time: %s, Magnitude: %s, Type: %s, Status: %s, RMS: %s' %
              (event_dict['id'], event_dict['time'], event_dict['mag'],
               event_dict['type'], event_dict['status'], event_dict['rms']))

        eq_times.append(UTCDateTime(event_dict['time']))
        eq_mags.append(float(event_dict['mag']))
        eq_depths.append(float(event_dict['depth']))
        eq_lats.append(float(event_dict['latitude']))
        eq_lons.append(float(event_dict['longitude']))

print('There are %d earthquakes in the requested catalog.\n' % count)


#%% retrieve phase picks

if PICK_QUERY:
    # Initialize arrays for phase information
    eq_picks = np.empty(len(evids), dtype=object)
    eq_phase_times = np.empty(len(evids), dtype=object)
    all_phases = []

    stations = STATION.split(',')

    for i, evid in enumerate(evids):
        eq_picks[i] = []
        eq_phase_times[i] = {}

        try:
            # Retrieve detailed event information
            detail = get_event_by_id(evid)

            # Retrieve phase information
            phases = get_phase_dataframe(detail, catalog=contributor)

            if phases is None or phases.empty:
                print('Event ID %s: No phase picks available' % evid)
                continue

            # Add event information
            phases['event_id'] = evid
            phases['origin_time'] = str(eq_times[i])
            phases['magnitude'] = eq_mags[i]

            all_phases.append(phases)

            # Find picks matching requested stations and channel
            picks = []
            phase_times = {}

            for _, pick in phases.iterrows():
                channel_id = str(pick['Channel']).split('.')

                if len(channel_id) < 3:
                    continue

                network = channel_id[0]
                station = channel_id[1]
                channel = channel_id[2]

                if station in stations and channel == CHANNEL and network == NETWORK:
                    picks.append(station)

                    # Store phase arrival times for each station
                    phase = str(pick['Phase'])
                    arrival_time = UTCDateTime(pick['Arrival Time'])

                    if station not in phase_times:
                        phase_times[station] = {}

                    if phase not in phase_times[station]:
                        phase_times[station][phase] = []

                    phase_times[station][phase].append(arrival_time)

            eq_picks[i] = picks
            eq_phase_times[i] = phase_times

            print('Event ID %s: %d phase picks, %d matching requested stations' %
                  (evid, len(phases), len(picks)))

        except Exception as e:
            print('Error retrieving picks for event %s: %s' % (evid, e))

    # Combine phase information into a single DataFrame
    if len(all_phases) > 0:
        phase_df = pd.concat(all_phases, ignore_index=True)

        print('Retrieved %d total phase picks.' % len(phase_df))
        #print(phase_df.head())

    else:
        phase_df = pd.DataFrame()
        print('No phase picks retrieved.')


eq_times = np.array(eq_times)
eq_mags = np.array(eq_mags)
eq_depths = np.array(eq_depths)
eq_lats = np.array(eq_lats)
eq_lons = np.array(eq_lons)


#%% read in the data and plot the events

client = Client(SOURCE)
inv = client.get_stations(starttime=starttime, endtime=endtime,
                          network=NETWORK, station=STATION,
                          location=LOCATION, channel=CHANNEL,
                          level="response")

ntimes = len(eq_times)

for sta in STATION.split(','):
    print(f'\nRunning for {sta}')

    eq_times_actual = []
    evids_actual = []
    eq_mags_actual = []
    eq_depths_actual = []
    start_actual = []
    end_actual = []
    pick = []
    st = Stream()
    cft_max = []
    phase_times_actual = []
    p_pick_times_actual = []

    for i in range(ntimes):

        if PICK_QUERY:
            # Check whether station has phase picks
            station_picks = eq_phase_times[i].get(sta, {})

            p_arrivals = [arrival
                          for phase, arrivals in station_picks.items()
                          if phase.upper().startswith('P')
                          for arrival in arrivals]

            if not p_arrivals:
                print(f'Station {sta} has no P pick for {eq_times[i]}; skipping')
                continue

            # Use the first P-family arrival (P, Pg, Pn, etc.) as time zero.
            p_pick_time = min(p_arrivals)
            pick_tmp = 'Y'
        else:
            raise RuntimeError('PICK_QUERY must be True to align waveforms by P pick')

        print(f'Getting data for {eq_times[i]} aligned to P pick {p_pick_time}')

        start_tmp = p_pick_time - PRE_WIN
        end_tmp = p_pick_time + DUR - PRE_WIN

        try:
            st_tmp = client.get_waveforms(NETWORK, sta, LOCATION, CHANNEL,
                                          starttime=start_tmp-PAD,
                                          endtime=end_tmp+PAD)

            if not st_tmp:
                print('No waveform data.')
                continue

            st_tmp.merge(method=1, fill_value=0)

            # filter
            st_tmp.detrend(type='linear')
            st_tmp.taper(max_percentage=None, max_length=5)
            st_tmp.filter('highpass', freq=FILT, corners=2, zerophase=True)

            if len(st_tmp) > 1:
                print('Multiple traces returned, using first trace.')

            tr_tmp = st_tmp[0]

            if tr_tmp.stats.endtime - tr_tmp.stats.starttime < DUR + PAD*2 - 1:
                print('Skipping due to short trace.')
                continue

            if tr_tmp.stats.sampling_rate != np.round(tr_tmp.stats.sampling_rate):
                print(f'Non-integer sampling rate: {tr_tmp.stats.sampling_rate}, rounding.')

            st_tmp.trim(start_tmp-PAD, end_tmp+PAD, pad=True, fill_value=0)

            # Run STA/LTA to cull out high SNR stations
            stf_cut = st_tmp.copy()
            df = stf_cut[0].stats.sampling_rate

            #stf_cut.detrend(type='linear')
            #stf_cut.taper(max_percentage=None, max_length=5)
            #stf_cut.filter('bandpass', freqmin=FILT[0], freqmax=FILT[1],
            #               corners=2, zerophase=True)

            tr = stf_cut[0]
            tr.trim(start_tmp, end_tmp)

            cft = classic_sta_lta(tr.data, int(STA * df), int(LTA * df))
            max_cft = np.max(cft)

            # # Skip high SNR values
            # if max_cft > STALTA_THRESH:
            #     print(f'Skipping due to high STA/LTA, {tr.id}, '
            #           f'{tr.stats.starttime}: {max_cft:.1f}')
            #     continue

            cft_max.append(max_cft)

            st.append(st_tmp[0].copy())
            start_actual.append(start_tmp)
            end_actual.append(end_tmp)
            eq_times_actual.append(eq_times[i])
            evids_actual.append(evids[i])
            eq_mags_actual.append(eq_mags[i])
            eq_depths_actual.append(eq_depths[i])
            phase_times_actual.append(station_picks)
            p_pick_times_actual.append(p_pick_time)
            pick.append(pick_tmp)

        except Exception as e:
            print(f'No data for {sta}, {eq_times[i]}: {e}')
            continue

    if not st:
        print(f'Error or no data found for {sta}')
        continue

    # Remove instrument response
    if CHANNEL == 'HHZ':
        st.remove_sensitivity(inventory=inv)
    else:
        st.remove_response(inventory=inv, water_level=60)

    #%% plot the waveforms
    stf = st.copy()
    stf.detrend(type='linear')
    stf.taper(max_percentage=None, max_length=5)
    stf.filter('highpass', freq=FILT, corners=2, zerophase=True)

    for i, tr in enumerate(stf):
        # Time relative to the station's P pick (P is at 0 s).
        t = tr.times() + (tr.stats.starttime - p_pick_times_actual[i])
        visible = (t >= -PRE_WIN) & (t <= DUR - PRE_WIN)
        if not np.any(visible):
            continue

        fig_wf, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t[visible], tr.data[visible], color='k', linewidth=.65)

        # Add event information
        ax.text(0.01, 0.98,
                f'M{eq_mags_actual[i]:.1f}  STA/LTA={cft_max[i]:.1f}',
                transform=ax.transAxes, va='top', fontsize=9)

        # Plot phase arrivals
        if PICK_QUERY:
            for phase, arrivals in phase_times_actual[i].items():
                for arrival in arrivals:
                    pick_time = arrival - p_pick_times_actual[i]

                    if -PRE_WIN <= pick_time <= DUR - PRE_WIN:
                        if phase.upper().startswith('P'):
                            color = 'r'
                        elif phase.upper().startswith('S'):
                            color = 'b'
                        else:
                            color = 'g'

                        ax.axvline(pick_time, color=color, linewidth=1.5)

        ax.set_xlim(-PRE_WIN, DUR - PRE_WIN)
        ax.set_ylabel('Ground velocity [m/s]')
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.set_xlabel('Time relative to P pick [s]')
        ax.set_title(
            f'{NETWORK}.{sta}.{CHANNEL} - {evids_actual[i]} - '
            f'{eq_times_actual[i].strftime("%Y-%m-%d %H:%M:%S")}'
        )

        plt.tight_layout()

        if SAVE:
            dt_fmt = p_pick_times_actual[i].datetime.strftime('%Y%m%dT%H%M%S')
            fig_wf.savefig(f'{SVDIR}/{tr.id}_{dt_fmt}_wf.png', dpi=250)

        # This blocking call advances to the next event after the figure closes.
        plt.show()
