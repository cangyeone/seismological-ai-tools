from obspy.clients.filesystem.tsindex import Client
from obspy import UTCDateTime
class OBSPYData():
    def __init__(self):
        self.client = Client("timeseries.sqlite") 
    def fetch_data(self):
        t = UTCDateTime("2019-07-26T00:00:00.019500")
        st = self.client.get_waveforms("X1", "51054", "01", "*", t, t + 3)
        print(len(st), st)
if __name__ == "__main__":
    data = OBSPYData() 
    get = data.fetch_data()
    print(get)