import sqlite3 
from collections import namedtuple
from .miniseed import MiniseedDataExtractor 
class DBSData():
    def __init__(self):
        self.conn = sqlite3.connect("timeseries.sqlite", 10.0)
        self.mseedextrc = MiniseedDataExtractor()
    def fetch_index_rows(self, req=None):
        cur = self.conn.cursor()
        request_table = "request_temp"
        cur.execute("CREATE TEMPORARY TABLE {0} "
                "(network TEXT, station TEXT, location TEXT, channel TEXT, "
                    "starttime TEXT, endtime TEXT) ".format(request_table))
        
        cur.execute("INSERT INTO {0} (network,station,location,channel,starttime,endtime) "
                                "VALUES (?,?,?,?,?,?) ".format(request_table), req)
        index_table = "tsindex"
        summary_table = "{0}_summary".format(index_table)
        cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' and name='{0}'".format(summary_table))
        summary_present = cur.fetchone()[0]
        cur.execute("UPDATE {0} SET starttime='0000-00-00T00:00:00' WHERE starttime='*'".format(request_table))
        cur.execute("UPDATE {0} SET endtime='5000-00-00T00:00:00' WHERE endtime='*'".format(request_table))
        sql = ("SELECT DISTINCT ts.network,ts.station,ts.location,ts.channel,ts.quality, "
                    "ts.starttime,ts.endtime,ts.samplerate, "
                    "ts.filename,ts.byteoffset,ts.bytes,ts.hash, "
                    "ts.timeindex,ts.timespans,ts.timerates, "
                    "ts.format,ts.filemodtime,ts.updated,ts.scanned, r.starttime, r.endtime "
                    "FROM {0} ts, {1} r "
                    "WHERE "
                    "  ts.network {2} r.network "
                    "  AND ts.station {2} r.station "
                    "  AND ts.location {2} r.location "
                    "  AND ts.channel {2} r.channel "
                    "  AND ts.starttime <= r.endtime "
                    "  AND ts.starttime >= datetime(r.starttime,'-{3} days') "
                    "  AND ts.endtime >= r.starttime "
                    .format(index_table, request_table, "=", 1))
        cur.execute(sql)
        
            # Map raw tuples to named tuples for clear referencing
        NamedRow = namedtuple('NamedRow',
                                ['network', 'station', 'location', 'channel', 'quality',
                                'starttime', 'endtime', 'samplerate', 'filename',
                                'byteoffset', 'bytes', 'hash', 'timeindex', 'timespans',
                                'timerates', 'format', 'filemodtime', 'updated', 'scanned',
                                'requeststart', 'requestend'])

        index_rows = []
        while True:
            row = cur.fetchone()
            if row is None:
                break
            index_rows.append(NamedRow(*row))

        index_rows.sort()

        cur.execute("DROP TABLE {0}".format(request_table))
        #conn.close()

        return index_rows
    def fetch_data(self, req, file_name):
        total_bytes = 0
        src_bytes = {}
        outfile = open(file_name, "wb")
        idx_row = self.fetch_index_rows(req)
        for data_segment in self.mseedextrc.extract_data(idx_row):
            shipped_bytes = data_segment.get_num_bytes()
            src_name = data_segment.get_src_name()
            if shipped_bytes > 0:
                data_segment.write(outfile)
                total_bytes += shipped_bytes
                src_bytes.setdefault(src_name, 0)
                src_bytes[src_name] += shipped_bytes
