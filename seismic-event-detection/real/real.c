/****************************************************************************
 *REAL - Rapid Earthquake Association and Location
 *
 *What you need:
 *  1. Traveltime table for P or/and S waves (dist,dep,P arrival,S arrival ...)
 *  2. Station information (stlo,stla,net,sta,chan,elev)
 *  3. Picks at each station and their weight and amplitude
 *  4. Control parameters (see usage)
 *      a. searched range and grid size
 *      b. average velocities of P and S waves
 *      c. date of the day
 *      d. thresholds
 *
 *Output:
 *  1. Associated and located earthquakes with origin time, magnitude, and
 *location
 *  2. Associated picks for each earthquake
 *  (local magnitude is preliminarily estimated based on HUTTON and BOORE, BSSA,
 *1987)
 *
 *Usage:
 *  See usage as below
 *
 *Author:
 *  Miao Zhang, Stanford University
 *  Now at Dalhousie University (miao.zhang@dal.ca)
 *
 *Reference:
 *  Miao Zhang, William Ellsworth and Greg Beroza, Rapid Earthquake Association
 *and Location, 2019 https://doi.org/10.1785/0220190052
 *
 *Revision history:
 *  June     2018       M. Zhang    Initial version in C
 *  June     2019	    M. Zhang    Release version 1.0
 ************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PHASESEL "phase_sel.txt"
#define CATALOGSEL "catalog_sel.txt"
#define RESOLUTION "resolution.txt"

//#define MAXTIME 86400.00 //one day
#define MAXTIME 2700000.00 // one month

typedef struct ttable {
  float gdist;
  float dep;
  float ptime;
  float stime;
  float prayp;
  float srayp;
  float phslow;
  float shslow;
  char pphase[10];
  char sphase[10];
} TTT;

typedef struct reselect {
  int num1;
  char otime1[50];
  float atime1;
  float std1;
  float lat1;
  float lon1;
  float dep1;
  int nofp1;
  int nofs1;
  int ntotal1;
  int nofps1;
} SELECT;

typedef struct picks {
  char net[5];
  char sta[8];
  char phase[5];
  float abs_pk;
  float pk;
  float amp;
  float res;
  float baz;
  float weig;
  float mag;
} PICK;

typedef struct clearups {
  char otime[50];
  float atime;
  float std;
  float lat;
  float lon;
  float dep;
  float mag_median;
  float mag_std;
  int pcount;
  int scount;
  int pscount;
  int psboth;
  float gap;
  PICK *pk;
} CLEARUP;

typedef struct trigg {
  float trig;
  float weight;
  float amp;
} TRIG;

typedef struct stationinfo {
  float stlo;
  float stla;
  char net[5];
  char sta[8];
  char comp[4];
  float elev;
} STATION;

void ddistaz(float, float, float, float, float *, float *);
float CalculateMedian(float *, int);
float CalculateMean(float *, int);
float CalculateStd(float *, float, int);
float Find_min(float **, int, int);
float Find_max(float **, int, int);
void Find_min_loc(float **, int, int, float *, int *, int *);
int Readttime(char *, TTT *, int);
int Readstation(char *, STATION *, int);
int DetermineNp(float **, int, int);
int DetermineNg(TRIG **, TRIG **, int, int);
void SortTriggers0(TRIG **, TRIG **, float **, float **, float **, float **,
                   float **, float **, int, int);
void DeleteOne(float **, int, int, int);
int DetermineNprange(float **, float, int, int);
void DetermineNps0range(float **, float **, float, float, float, float,
                        int, int);
int ReselectFinal(SELECT *, int);
void ReselectClear(CLEARUP *, int);
void Accounttriggers_homo(float, float, float, float, float, float, int);
void Accounttriggers_layer(float, float, float, float, float, float, int);
void Sortpscounts(float **, int);

// global
float tint;
float **pscounts;
STATION *ST;
TRIG **TGP, **TGS;
TTT *TB;
float ptw, stw, nrt;
float **ptrig, **temp, **ptrig0, **strig0;
float vp0, vs0, s_vp0, s_vs0;
int NNps, Nps2;
int np0, ns0, nps0, npsboth0;
int *np0_start, *np0_end, *ns0_start, *ns0_end;
float tpmin0, tdx, tdh, trx, trh;
float dtps;
float GAPTH;
float GCarc0;
float std0;

int Nst = 1000;   // maximum number of stations
int Nps = 20000; // maximum number of P/S picks recorded at one station
int Ntb = 20000; // maximum number of lines in traveltime table
int ispeed = 1;  // default setting ispeed = 1

// main function
int main(int argc, char **argv) {
  int i, j, k, l, m, n;
  FILE *fp, *fpr, *fp1, *fp2;
  char output1[256], output2[256], dir[256], input[256];
  int test, error, pcount, scount, psboth, puse, nnn, ps, nselect;
  float dx, dh, rx, rh, dx1, dx2, rx1, rx2;
  float tp0_cal, ts0_cal, tp_cal, ts_cal, tp_pre, ts_pre, tp_pre_b, ts_pre_b,
      tp_pre_e, ts_pre_e;
  float median, std, GCarc, rdist, distmax, baz;
  float told, lonref, latref, elevref, latref0, lonref0;
  float lonmin, lonmax, latmin, latmax, lat0, lon0, dep, latcenter;
  float stlamin, stlamax, stlomin, stlomax;
  int ttd, tth, tts, mmm;
  int nlat, nlon, ndep;
  float ttm, ptemp, rsel;
  char otime[50];
  int igrid, ires, ielev, ig, ih, im, iremove, inoref;
  SELECT *RELC;
  CLEARUP *CLEAR;
  float **pamp0, **samp0, **pweight0, **sweight0, *mag;
  float mag_median, mag_std, p_mag, s_mag;
  int nyear, nmon, nday;
  float tpmin, tpmax, tsmin, tsmax, Maxt0;

  // initiating parameters
  error = 0;
  igrid = 0;
  ielev = 0;
  ires = 0;
  // rsel*STD
  rsel = 5;
  latref0 = -10000;
  lonref0 = -10000;
  s_vp0 = 1000000;
  s_vs0 = 1000000;
  // station azimuth gap threshold (default: no constraint)
  GAPTH = 360;
  // only use picks within GCarc0 (in degree) (default: no constraint)
  GCarc0 = 180;
  // avoid using fixed multiplication of 111.19 km/deg, change with your
  // latcenter (suggested by Ruijia Wang)
  latcenter = 0.0;

  for (i = 1; !error && i < argc; i++) {
    if (argv[i][0] == '-') {
      switch (argv[i][1]) {
      case 'R':
        sscanf(&argv[i][2], "%f/%f/%f/%f/%f/%f/%f/%f/%f", &rx, &rh,
               &dx, &dh, &tint, &GAPTH, &GCarc0, &latref0, &lonref0);
        break;
      case 'S':
        sscanf(&argv[i][2], "%d/%d/%d/%d/%f/%f/%f/%f/%d", &np0, &ns0, &nps0,
               &npsboth0, &std0, &dtps, &nrt, &rsel, &ires);
        break;
      case 'V':
        sscanf(&argv[i][2], "%f/%f/%f/%f/%d", &vp0, &vs0, &s_vp0, &s_vs0,
               &ielev);
        break;
      case 'G':
        sscanf(&argv[i][2], "%f/%f/%f/%f", &trx, &trh, &tdx, &tdh);
        igrid = 1;
        break;
      case 'D':
        sscanf(&argv[i][2], "%d/%d/%d/%f", &nyear, &nmon, &nday, &latcenter);
        break;
      default:
        error = 1;
        break;
      }
    }
  }

  // Usage
  if (argc < 3 || error == 1) {
    fprintf(stderr, "Usage:  Rapid Earthquake Association and Location "
                    "(REAL, May 2021 version)\n");
    fprintf(stderr, "   -D(nyear/nmon/nday/lat_center) "
                    "-R(rx/rh/tdx/tdh/tint[/gap/GCarc0/latref0/lonref0]]) "
                    "-V(vp0/vs0/[s_vp0/s_vs0/ielev])\n");
    fprintf(stderr, "   -S(np0/ns0/nps0/npsboth0/std0/dtps/nrt[/rsel/ires]) "
                    "[-G(trx/trh/tdx/tdh)] station pickdir [ttime]\n");
    fprintf(stderr, "   "
                    "------------------------------------explanation-----------"
                    "---------------------------------\n");
    fprintf(stderr,
            "   -D: date of the day (year/month/day) and latitude center "
            "(deg., so that lat and lon have consistent distance in km)\n");
    fprintf(stderr,
            "   -R: search ranges and grids around the station that recorded "
            "initiating pick in horizontal direction and depth,\n");
    fprintf(stderr,
            "       event interval, largest station gap, largest distance, "
            "reference location (deg/km/deg/km/sec[deg/deg/deg/deg])\n");
    fprintf(stderr, "   -V: average velocities and near-surface velocities of "
                    "P and S waves, station elevation_or_not\n");
    fprintf(stderr, "       (km/s|km/s|[km/s|km/s|int])\n");
    fprintf(stderr, "   -S: thresholds: number of picks (P,S,P+S), number of "
                    "stations with both P and S,STD threshold, \n");
    fprintf(stderr, "       S-P interval,nrt*length of time window, only keep "
                    "picks with residuals < rsel*STD,\n");
    fprintf(stderr, "       resolution_or_not "
                    "(int/int/int/int/float/float/float/[float/int])\n");
    fprintf(stderr, "   -G: range and grid settings in traveltime table (in "
                    "horizontal and vertical) (deg/km/deg/km)\n");
    fprintf(stderr, "   station: station information; pickdir: directory of "
                    "picks; ttime: [traveltime table]\n");
    exit(-1);
  }

  fprintf(stderr, "Max Setting: Nst %-5d Nps %-5d Ntb %-5d\n", Nst, Nps, Ntb);

  /* read station information */
  if (igrid == 0) {
    strcpy(input, argv[5]);
  } else {
    strcpy(input, argv[6]);
  }
  ST = (STATION *)malloc(sizeof(STATION) * Nst);
  Nst = Readstation(input, ST, Nst);

  if (ielev == 0) {
    for (i = 0; i < Nst; i++)
      ST[i].elev = 0.0;
  }

  stlamin = 1.0e8;
  stlomin = 1.0e8;
  stlamax = -1.0e8;
  stlomax = -1.0e8;
  for (i = 0; i < Nst; i++) {
    if (ST[i].stla > stlamax)
      stlamax = ST[i].stla;
    if (ST[i].stlo > stlomax)
      stlomax = ST[i].stlo;
    if (ST[i].stla < stlamin)
      stlamin = ST[i].stla;
    if (ST[i].stlo < stlomin)
      stlomin = ST[i].stlo;
  }
  ddistaz(stlamin, stlomin, stlamax, stlomax, &distmax, &baz);

  /* read triggers */
  if (igrid == 0) {
    strcpy(dir, argv[6]);
  } else {
    strcpy(dir, argv[7]);
  }

  TGP = (TRIG **)malloc(sizeof(TRIG *) * Nst);
  TGS = (TRIG **)malloc(sizeof(TRIG *) * Nst);
  for (i = 0; i < Nst; i++) {
    TGP[i] = (TRIG *)malloc(sizeof(TRIG) * Nps);
    TGS[i] = (TRIG *)malloc(sizeof(TRIG) * Nps);
  }

  for (i = 0; i < Nst; i++) {
    for (j = 0; j < Nps; j++) {
      TGP[i][j].trig = 1.0e8;
      TGP[i][j].weight = 0.0;
      TGP[i][j].amp = 0.0;
      TGS[i][j].trig = 1.0e8;
      TGS[i][j].weight = 0.0;
      TGS[i][j].amp = 0.0;
    }
  }

  for (i = 0; i < Nst; i++) {
    sprintf(input, "%s/%s.%s.P.txt", dir, ST[i].net, ST[i].sta);
    if ((fp = fopen(input, "r")) == NULL) {
      fprintf(stderr, "Can not open file in ReadFile %s\n", input);
    } else {
      test = 0;
      for (j = 0; j < Nps; j++) {
        if (fscanf(fp, "%f %f %f", &TGP[i][j].trig, &TGP[i][j].weight,
                   &TGP[i][j].amp) == EOF)
          test = 1;
        if (TGP[i][j].trig > MAXTIME)
          TGP[i][j].trig = 1.0e8;
        if (test == 1)
          break;
      }
      fclose(fp);
    }

    sprintf(input, "%s/%s.%s.S.txt", dir, ST[i].net, ST[i].sta);
    if ((fp = fopen(input, "r")) == NULL) {
      fprintf(stderr, "Can not open file in ReadFile %s\n", input);
    } else {
      test = 0;
      for (j = 0; j < Nps; j++) {
        if (fscanf(fp, "%f %f %f", &TGS[i][j].trig, &TGS[i][j].weight,
                   &TGS[i][j].amp) == EOF)
          test = 1;
        if (TGS[i][j].trig > MAXTIME)
          TGS[i][j].trig = 1.0e8;
        if (test == 1)
          break;
      }
      fclose(fp);
    }
  }

  /* read travel time table */
  if (igrid == 1) {
    strcpy(input, argv[8]);
    if ((TB = malloc(sizeof(TTT) * Ntb)) == NULL) {
      fprintf(stderr, "malloc memory error for TTT\n");
      exit(-1);
    }
    Ntb = Readttime(input, TB, Ntb);
  }

  Nps = DetermineNg(TGP, TGS, Nst, Nps);
  NNps = Nps;

  dx2 = dx / cos(latcenter * 3.1415926 / 180.0);
  rx2 = rx / cos(latcenter * 3.1415926 / 180.0);
  dx1 = dx;
  rx1 = rx;
  fprintf(stderr, "Actual     : Nst %-5d Nps %-5d Ntb %-5d\n", Nst, Nps - 1,
          Ntb);
  if (latref0 > -999 && lonref0 > -999) {
    fprintf(stderr, "searching range: %f %f %f %f\n", latref0 - rx1,
            latref0 + rx1, lonref0 - rx2, lonref0 + rx2);
    fprintf(stderr, "                 lat_range=+-%f lon_range=+-%f\n", rx1,
            rx2);
    fprintf(stderr, "                 lat_grid=%f lon_grid=%f\n", dx1, dx2);
  } else {
    fprintf(stderr, "searching range: %f %f %f %f\n", stlamin, stlamax,
            stlomin, stlomax);
    fprintf(stderr, "                 lat_range=+-%f lon_range=+-%f\n", rx1,
            rx2);
    fprintf(stderr, "                 lat_grid=%f lon_grid=%f\n", dx1, dx2);
  }

  ptrig = (float **)malloc(sizeof(float *) * Nst);
  ptrig0 = (float **)malloc(sizeof(float *) * Nst);
  strig0 = (float **)malloc(sizeof(float *) * Nst);
  pamp0 = (float **)malloc(sizeof(float *) * Nst);
  samp0 = (float **)malloc(sizeof(float *) * Nst);
  pweight0 = (float **)malloc(sizeof(float *) * Nst);
  sweight0 = (float **)malloc(sizeof(float *) * Nst);
  temp = (float **)malloc(sizeof(float *) * Nst);
  for (i = 0; i < Nst; i++) {
    ptrig[i] = (float *)malloc(sizeof(float) * Nps);
    ptrig0[i] = (float *)malloc(sizeof(float) * Nps);
    strig0[i] = (float *)malloc(sizeof(float) * Nps);
    pamp0[i] = (float *)malloc(sizeof(float) * Nps);
    samp0[i] = (float *)malloc(sizeof(float) * Nps);
    pweight0[i] = (float *)malloc(sizeof(float) * Nps);
    sweight0[i] = (float *)malloc(sizeof(float) * Nps);
    temp[i] = (float *)malloc(sizeof(float) * Nps);
  }

  // default number of events (picks*Nst)
  RELC = (SELECT *)malloc(sizeof(SELECT) * Nst * Nps);
  CLEAR = (CLEARUP *)malloc(sizeof(CLEARUP) * Nst * Nps);
  for (i = 0; i < Nst * Nps; i++)
    CLEAR[i].pk = (PICK *)malloc(sizeof(PICK) * Nst * 2);
  /* determine traveltime across one grid*/
  // lon grid (dx2) has been corrected, consistent with lat grid (dx1)
  ptw = sqrt((dx1 * 111.19) * (dx1 * 111.19) + (dx1 * 111.19) * (dx1 * 111.19) +
             dh * dh) / vp0;
  stw = sqrt((dx1 * 111.19) * (dx1 * 111.19) + (dx1 * 111.19) * (dx1 * 111.19) +
             dh * dh) / vs0;

  if (tint < stw)
    tint = stw;
  fprintf(stderr,
          "p-window= %.2f sec; s-window= %.2f sec; event-window= %.2f sec\n",
          nrt * ptw, nrt * stw, tint);

  // sort triggers
  SortTriggers0(TGP, TGS, ptrig0, strig0, pamp0, samp0, pweight0, sweight0, Nst,
                Nps);
  for (i = 0; i < Nst; i++) {
    for (j = 0; j < Nps; j++) {
      ptrig[i][j] = ptrig0[i][j];
    }
  }

  nlat = (int)(2 * rx1 / dx1 + 1);
  nlon = (int)(2 * rx2 / dx2 + 1);
  ndep = (int)(rh / dh + 1);
  nnn = nlat * nlon * ndep;
  printf("Nlat= %d Nlon= %d Ndep= %d\n", nlat, nlon, ndep);

  pscounts = (float **)malloc(nnn * sizeof(float *));
  for (k = 0; k < nnn; k++) {
    pscounts[k] = (float *)malloc(10 * sizeof(float));
  }

  np0_start = (int *)malloc(sizeof(int) * Nst);
  np0_end = (int *)malloc(sizeof(int) * Nst);
  ns0_start = (int *)malloc(sizeof(int) * Nst);
  ns0_end = (int *)malloc(sizeof(int) * Nst);

  told = 0.0;
  mmm = 0;
  m = 0;

  inoref = -1;
  if (latref0 < -999 && lonref0 < -999)
    inoref = 1;
  Maxt0 = Find_max(ptrig, Nst, Nps);
  // search each initiating P pick
  while (Find_min(ptrig, Nst, Nps) < Maxt0) {
    // jump:
    Nps = DetermineNp(ptrig, Nst, Nps);
    Find_min_loc(ptrig, Nst, 1, &tpmin0, &m, &n);
    if (fabs(tpmin0 - 1.0e8) < 1)
      break;

    lonref = ST[m].stlo;
    latref = ST[m].stla;
    elevref = ST[m].elev;
    if (inoref > 0) {
      lonref0 = ST[m].stlo;
      latref0 = ST[m].stla;
    }

    // Make sure you know what you are doing!
    // if(fabs(tpmin0 - told) < tint/2){
    //	DeleteOne(ptrig,m,Nps,n);
    //	goto jump;
    //}

    tpmin = tpmin0 - 1.2 * (distmax * 111.19 / vp0);
    tpmax = tpmin0 + 1.2 * (distmax * 111.19 / vp0);
    tsmin = tpmin0 - 1.2 * (distmax * 111.19 / vs0);
    tsmax = tpmin0 + 1.2 * (distmax * 111.19 / vs0);

    Nps2 = DetermineNprange(ptrig, tpmax, Nst, Nps);
    // printf("%d %f %f\n",Nps,told,tpmin0);

    if (tpmin < 0.0)
      tpmin = 0.0;
    if (tsmin < 0.0)
      tsmin = 0.0;
    if (tpmax > MAXTIME)
      tpmax = MAXTIME;
    if (tsmax > MAXTIME)
      tsmax = MAXTIME;

    DetermineNps0range(ptrig0, strig0, tpmin, tpmax, tsmin, tsmax, Nst, NNps);

    for (k = 0; k < nnn; k++) {
      for (l = 0; l < 10; l++) {
        pscounts[k][l] = 0.0;
      }
    }

    // homo model
    if (igrid == 0) {
#pragma omp parallel for shared(pscounts)                                      \
    firstprivate(latref, lonref, latref0, lonref0, elevref, nlon, ndep, dx1,   \
                 dx2, dh) private(lat0, lon0, dep, l, i, j, k)
      for (l = 0; l < nnn; ++l) {
        i = (int)(l / (nlon * ndep));
        j = (int)((l - i * nlon * ndep) / ndep);
        k = l - i * nlon * ndep - j * ndep;
        // In case that searched location is co-located with the station
        // position (gcarc == 0).
        lat0 = latref0 - rx1 + i * dx1 + 0.01234 * dx1;
        lon0 = lonref0 - rx2 + j * dx2 + 0.01234 * dx2;
        dep = k * dh;
        Accounttriggers_homo(lat0, lon0, dep, latref, lonref, elevref, l);
      }
#pragma omp barrier
      // layer model
    } else {
#pragma omp parallel for shared(pscounts)                                      \
    firstprivate(latref, lonref, latref0, lonref0, elevref, nlon, ndep, dx1,   \
                 dx2, dh) private(lat0, lon0, dep, l, i, j, k)
      for (l = 0; l < nnn; ++l) {
        i = (int)(l / (nlon * ndep));
        j = (int)((l - i * nlon * ndep) / ndep);
        k = l - i * nlon * ndep - j * ndep;
        // In case that searched location is co-located with the station
        // position (gcarc == 0).
        lat0 = latref0 - rx1 + i * dx1 + 0.01234 * dx1;
        lon0 = lonref0 - rx2 + j * dx2 + 0.01234 * dx2;
        dep = k * dh;
        Accounttriggers_layer(lat0, lon0, dep, latref, lonref, elevref, l);
      }
#pragma omp barrier
    }
    // only output the resolution file for the first effective event (the first
    // pick should be true)
    if (ires == 1) {
      fpr = fopen(RESOLUTION, "w");
      for (k = 0; k < nnn; k++) {
        fprintf(fpr, "%12.4lf %12.4lf %12.4lf %12.4lf %4d %4d %4d %8.4lf\n",
                pscounts[k][3], pscounts[k][0], pscounts[k][1], pscounts[k][2],
                (int)pscounts[k][4], (int)pscounts[k][5], (int)pscounts[k][7],
                pscounts[k][6]);
      }
      fclose(fpr);
      exit(-1);
    }

    // sort pscounts
    Sortpscounts(pscounts, nnn);

    if (pscounts[nnn - 1][4] >= np0 && pscounts[nnn - 1][5] >= ns0 &&
        pscounts[nnn - 1][7] >= nps0 && pscounts[nnn - 1][6] <= std0 &&
        pscounts[nnn - 1][8] <= GAPTH && pscounts[nnn - 1][9] >= npsboth0) {
      told = pscounts[nnn - 1][3];
      ttd = (int)(pscounts[nnn - 1][3] / 86400);
      tth = (int)((pscounts[nnn - 1][3] - ttd * 86400) / 3600);
      tts = (int)((pscounts[nnn - 1][3] - ttd * 86400 - tth * 3600) / 60);
      ttm = pscounts[nnn - 1][3] - ttd * 86400 - tth * 3600 - tts * 60;
      sprintf(otime, "%04d %02d %02d %02d:%02d:%06.3f", nyear, nmon, ttd + nday,
              tth, tts, ttm);

      RELC[mmm].num1 = mmm + 1;
      strcpy(RELC[mmm].otime1, otime);
      RELC[mmm].atime1 = pscounts[nnn - 1][3];
      RELC[mmm].std1 = pscounts[nnn - 1][6];
      RELC[mmm].lat1 = pscounts[nnn - 1][0];
      RELC[mmm].lon1 = pscounts[nnn - 1][1];
      RELC[mmm].dep1 = pscounts[nnn - 1][2];
      RELC[mmm].nofp1 = pscounts[nnn - 1][4];
      RELC[mmm].nofs1 = pscounts[nnn - 1][5];
      RELC[mmm].ntotal1 = pscounts[nnn - 1][7];
      RELC[mmm].nofps1 = pscounts[nnn - 1][9];

      fprintf(stderr,
              "%5d %25s %12.3lf %8.4lf %12.4lf %12.4lf %12.4lf %4d %4d %4d %4d "
              "%8.2f\n",
              mmm + 1, otime, pscounts[nnn - 1][3], pscounts[nnn - 1][6],
              pscounts[nnn - 1][0], pscounts[nnn - 1][1], pscounts[nnn - 1][2],
              (int)(pscounts[nnn - 1][4]), (int)(pscounts[nnn - 1][5]),
              (int)(pscounts[nnn - 1][7]), (int)(pscounts[nnn - 1][9]),
              pscounts[nnn - 1][8]);
      mmm++;

      iremove = 0;
      // Ispeed is recommended to save time (use a strict threshold)
      // Otherwise, one event would be associated and located by many initiating
      // P picks. That's huge!
      if (ispeed > 1.0e-5) {
        for (k = 0; k < Nst; k++) {
          lat0 = pscounts[nnn - 1][0];
          lon0 = pscounts[nnn - 1][1];
          dep = pscounts[nnn - 1][2];

          ddistaz(ST[k].stla, ST[k].stlo, lat0, lon0, &GCarc, &baz);
          if (igrid == 0) {
            tp_cal =
                sqrt((GCarc * 111.19) * (GCarc * 111.19) + dep * dep) / vp0 +
                ST[k].elev / s_vp0;
          } else {
            ih = rint(dep / tdh);
            ig = ih * rint(trx / tdx) + rint(GCarc / tdx);
            tp_cal = TB[ig].ptime + (GCarc - TB[ig].gdist) * TB[ig].prayp +
                     (dep - TB[ig].dep) * TB[ig].phslow + ST[k].elev / s_vp0;
          }

          tp_pre = pscounts[nnn - 1][3] + tp_cal;
          tp_pre_b = tp_pre - nrt * ptw / 2.0;
          tp_pre_e = tp_pre + nrt * ptw / 2.0;

          if (tp_pre_b < 0.0)
            tp_pre_b = 0.0;
          if (tp_pre_e > MAXTIME)
            tp_pre_e = MAXTIME;

          // To speed up, remove those associated P picks
          for (j = 0; j < Nps2; j++) {
            if (ptrig[k][j] > tp_pre_b && ptrig[k][j] < tp_pre_e) {
              DeleteOne(ptrig, k, Nps, j);
              iremove++;
              break;
            }
          }
        }
      }
      // make sure the current initiating P is removed
      if (iremove < 1.0e-5) {
        DeleteOne(ptrig, m, Nps, n);
      }
    } else {
      DeleteOne(ptrig, m, Nps, n);
    }
  }

  fp1 = fopen(CATALOGSEL, "w");
  fp2 = fopen(PHASESEL, "w");
  /*Reselect to keep the most reliable event within a time window*/
  nselect = ReselectFinal(RELC, mmm);
  fprintf(stderr, "before first selection: %d\n after first selection: %d\n",
          mmm, nselect);

  mag = (float *)malloc(Nst * sizeof(float));
  for (i = 0; i < nselect; i++) {
    pcount = 0;
    scount = 0;
    psboth = 0;
    ps = 0;
    im = 0;
    for (k = 0; k < Nst; k++) {
      mag[k] = -100;
      lat0 = RELC[i].lat1;
      lon0 = RELC[i].lon1;
      dep = RELC[i].dep1;

      ddistaz(ST[k].stla, ST[k].stlo, lat0, lon0, &GCarc, &baz);
      rdist = sqrt((GCarc * 111.19) * (GCarc * 111.19) + dep * dep);

      if (igrid == 0) {
        tp_cal = rdist / vp0 + ST[k].elev / s_vp0;
        ts_cal = rdist / vs0 + ST[k].elev / s_vs0;
      } else {
        ih = rint(dep / tdh);
        ig = ih * rint(trx / tdx) + rint(GCarc / tdx);
        tp_cal = TB[ig].ptime + (GCarc - TB[ig].gdist) * TB[ig].prayp +
                 (dep - TB[ig].dep) * TB[ig].phslow + ST[k].elev / s_vp0;
        ts_cal = TB[ig].stime + (GCarc - TB[ig].gdist) * TB[ig].srayp +
                 (dep - TB[ig].dep) * TB[ig].shslow + ST[k].elev / s_vs0;
      }

      tp_pre = RELC[i].atime1 + tp_cal;
      ts_pre = RELC[i].atime1 + ts_cal;

      // RELC[i].std1/2.0 consider the origin time uncertainty
      tp_pre_b = tp_pre - nrt * ptw / 2.0 - RELC[i].std1 / 2.0;
      tp_pre_e = tp_pre + nrt * ptw / 2.0 + RELC[i].std1 / 2.0;
      ts_pre_b = ts_pre - nrt * stw / 2.0 - RELC[i].std1 / 2.0;
      ts_pre_e = ts_pre + nrt * stw / 2.0 + RELC[i].std1 / 2.0;
      if (tp_pre_b < 0.0)
        tp_pre_b = 0.0;
      if (ts_pre_b < 0.0)
        ts_pre_b = 0.0;
      if (tp_pre_e > MAXTIME)
        tp_pre_e = MAXTIME;
      if (ts_pre_e > MAXTIME)
        ts_pre_e = MAXTIME;

      p_mag = -100;
      s_mag = -100;
      ptemp = -100;
      puse = 0;
      for (j = 0; j < NNps; j++) {
        // rsel*std to remove some picks with large residuals
        if (ptrig0[k][j] > tp_pre_b && ptrig0[k][j] < tp_pre_e &&
            fabs(ptrig0[k][j] - tp_pre) < rsel * RELC[i].std1 &&
            GCarc < GCarc0) {
          strcpy(CLEAR[i].pk[ps].net, ST[k].net);
          strcpy(CLEAR[i].pk[ps].sta, ST[k].sta);
          strcpy(CLEAR[i].pk[ps].phase, "P");
          CLEAR[i].pk[ps].abs_pk = ptrig0[k][j];
          CLEAR[i].pk[ps].pk = ptrig0[k][j] - RELC[i].atime1;
          CLEAR[i].pk[ps].amp = pamp0[k][j];
          CLEAR[i].pk[ps].res = ptrig0[k][j] - tp_pre;
          CLEAR[i].pk[ps].baz = baz;
          CLEAR[i].pk[ps].weig = pweight0[k][j];

          p_mag = log(pamp0[k][j]) / log(10) +
                  1.110 * log(rdist / 100) / log(10) + 0.00189 * (rdist - 100) +
                  3.0;
          CLEAR[i].pk[ps].mag = p_mag;

          pcount++;
          ps++;
          puse = 1;
          ptemp = ptrig0[k][j];
          break;
        }
      }

      // dtps: to remove some false S picks (they may be P picks but wrongly
      // identified as S picks, it happens!) rsel*std to remove some picks with
      // large residuals
      for (j = 0; j < NNps; j++) {
        if ((ts_pre - tp_pre) > dtps && fabs(ptemp - strig0[k][j]) > dtps &&
            strig0[k][j] > ts_pre_b && strig0[k][j] < ts_pre_e &&
            fabs(strig0[k][j] - ts_pre) < rsel * RELC[i].std1 &&
            GCarc < GCarc0) {
          strcpy(CLEAR[i].pk[ps].net, ST[k].net);
          strcpy(CLEAR[i].pk[ps].sta, ST[k].sta);
          strcpy(CLEAR[i].pk[ps].phase, "S");
          CLEAR[i].pk[ps].abs_pk = strig0[k][j];
          CLEAR[i].pk[ps].pk = strig0[k][j] - RELC[i].atime1;
          CLEAR[i].pk[ps].amp = samp0[k][j];
          CLEAR[i].pk[ps].res = strig0[k][j] - ts_pre;
          CLEAR[i].pk[ps].baz = baz;
          CLEAR[i].pk[ps].weig = sweight0[k][j];

          s_mag = log(samp0[k][j]) / log(10) +
                  1.110 * log(rdist / 100) / log(10) + 0.00189 * (rdist - 100) +
                  3.0;
          CLEAR[i].pk[ps].mag = s_mag;

          scount++;
          ps++;
          if (puse == 1)
            psboth++;
          break;
        }
      }
      // amplitudes recorded at nearest stations are usually unstable
      // if(GCarc*111.19 > 10 && (p_mag > -90 || s_mag > -90)){
      if (p_mag > -90 || s_mag > -90) {
        if (p_mag > s_mag) {
          mag[im] = p_mag;
          im++;
        } else {
          mag[im] = s_mag;
          im++;
        }
      }
    }

    if (im < 2) {
      mag_median = -100.0;
      mag_std = -100.0;
    } else {
      mag_median = CalculateMedian(mag, im);
      mag_std = CalculateStd(mag, mag_median, im);
    }

    strcpy(CLEAR[i].otime, RELC[i].otime1);
    CLEAR[i].atime = RELC[i].atime1;
    CLEAR[i].std = RELC[i].std1;
    CLEAR[i].lat = RELC[i].lat1;
    CLEAR[i].lon = RELC[i].lon1;
    CLEAR[i].dep = RELC[i].dep1;
    CLEAR[i].mag_median = mag_median; // may update in ReselectClear
    CLEAR[i].mag_std = mag_std;       // may update in ReselectClear
    CLEAR[i].pcount = pcount;         // may update in ReselectClear
    CLEAR[i].scount = scount;         // may update in ReselectClear
    CLEAR[i].pscount = ps;            // may update in ReselectClear
    CLEAR[i].psboth = psboth;         // may update in ReselectClear
    CLEAR[i].gap = -100;              // will update in ReselectClear
  }

  /*Reselect to remove unstable events with large gap and exclude one pick is
   * associated more than once*/
  ReselectClear(CLEAR, nselect);

  k = 0;
  for (i = 0; i < nselect; i++) {
    if (CLEAR[i].pcount >= np0 && CLEAR[i].scount >= ns0 &&
        CLEAR[i].pscount >= nps0 && CLEAR[i].std <= std0 &&
        CLEAR[i].gap <= GAPTH && CLEAR[i].psboth >= npsboth0) {
      k++;
      if (CLEAR[i].lon > 180) {
        CLEAR[i].lon = CLEAR[i].lon - 360;
      }
      if (CLEAR[i].lon < -180) {
        CLEAR[i].lon = CLEAR[i].lon + 360;
      } // suggested by Yukuan Chen
      fprintf(fp1,
              "%5d %25s %12.3lf %8.4lf %12.4lf %12.4lf %12.4lf %8.3lf %8.3lf "
              "%4d %4d %4d %4d %8.2lf\n",
              k, CLEAR[i].otime, CLEAR[i].atime, CLEAR[i].std, CLEAR[i].lat,
              CLEAR[i].lon, CLEAR[i].dep, CLEAR[i].mag_median, CLEAR[i].mag_std,
              CLEAR[i].pcount, CLEAR[i].scount, CLEAR[i].pscount,
              CLEAR[i].psboth, CLEAR[i].gap);
      fprintf(fp2,
              "%5d %25s %12.3lf %8.4lf %12.4lf %12.4lf %12.4lf %8.3lf %8.3lf "
              "%4d %4d %4d %4d %8.2lf\n",
              k, CLEAR[i].otime, CLEAR[i].atime, CLEAR[i].std, CLEAR[i].lat,
              CLEAR[i].lon, CLEAR[i].dep, CLEAR[i].mag_median, CLEAR[i].mag_std,
              CLEAR[i].pcount, CLEAR[i].scount, CLEAR[i].pscount,
              CLEAR[i].psboth, CLEAR[i].gap);
      for (j = 0; j < CLEAR[i].pscount; j++) {
        fprintf(fp2,
                "%5s %8s %5s %12.4lf %12.4lf %12.4e %12.4lf %12.4lf %12.4lf\n",
                CLEAR[i].pk[j].net, CLEAR[i].pk[j].sta, CLEAR[i].pk[j].phase,
                CLEAR[i].pk[j].abs_pk, CLEAR[i].pk[j].pk, CLEAR[i].pk[j].amp,
                CLEAR[i].pk[j].res, CLEAR[i].pk[j].weig, CLEAR[i].pk[j].baz);
      }
    }
  }

  fprintf(stderr, "before second selection: %d\n after second selection: %d\n",
          nselect, k);

  fclose(fp1);
  fclose(fp2);
  free(np0_start);
  free(np0_end);
  free(ns0_start);
  free(ns0_end);
  for (i = 0; i < Nst; i++) {
    free(ptrig[i]);
    free(ptrig0[i]);
    free(strig0[i]);
    free(pamp0[i]);
    free(samp0[i]);
    free(TGP[i]);
    free(TGS[i]);
  }
  for (i = 0; i < nnn; i++)
    free(pscounts[i]);
  free(pscounts);
  free(TGP);
  free(TGS);
  free(ptrig);
  free(ptrig0);
  free(strig0);
  free(pamp0);
  free(samp0);
  free(ST);
  free(RELC);
  free(CLEAR);
  free(TB);
  free(mag);
  return 0;
}

// 1. remove unstable events with large station gap
// 2. If one pick is associated with multiple events (although the possibility
// is very low if you have suitable parameter settings),
//   keep the pick with smallest individual traveltime residual (old version) or
//   keep the pick with the event that has more associated picks (prefered now)
void ReselectClear(CLEARUP *CLEAR, int NN) {
  int i, j, k, l, m, idx;
  float *mag0, *res0, res_median, gap0, gap;
  int pcount, scount, psboth;
  extern int np0, ns0, nps0, npsboth0;
  char net[5], sta[8], phase[5];
  float abs_pk, pk, amp, res, baz, weig, mag;
  // sort baz
  for (i = 0; i < NN; i++) {
    for (j = 0; j < CLEAR[i].pscount; j++) {
      for (k = j; k < CLEAR[i].pscount; k++) {
        if (CLEAR[i].pk[j].baz > CLEAR[i].pk[k].baz) {
          strcpy(net, CLEAR[i].pk[j].net);
          strcpy(sta, CLEAR[i].pk[j].sta);
          strcpy(phase, CLEAR[i].pk[j].phase);
          abs_pk = CLEAR[i].pk[j].abs_pk;
          pk = CLEAR[i].pk[j].pk;
          amp = CLEAR[i].pk[j].amp;
          res = CLEAR[i].pk[j].res;
          baz = CLEAR[i].pk[j].baz;
          weig = CLEAR[i].pk[j].weig;
          mag = CLEAR[i].pk[j].mag;

          strcpy(CLEAR[i].pk[j].net, CLEAR[i].pk[k].net);
          strcpy(CLEAR[i].pk[j].sta, CLEAR[i].pk[k].sta);
          strcpy(CLEAR[i].pk[j].phase, CLEAR[i].pk[k].phase);
          CLEAR[i].pk[j].abs_pk = CLEAR[i].pk[k].abs_pk;
          CLEAR[i].pk[j].pk = CLEAR[i].pk[k].pk;
          CLEAR[i].pk[j].amp = CLEAR[i].pk[k].amp;
          CLEAR[i].pk[j].res = CLEAR[i].pk[k].res;
          CLEAR[i].pk[j].baz = CLEAR[i].pk[k].baz;
          CLEAR[i].pk[j].weig = CLEAR[i].pk[k].weig;
          CLEAR[i].pk[j].mag = CLEAR[i].pk[k].mag;

          strcpy(CLEAR[i].pk[k].net, net);
          strcpy(CLEAR[i].pk[k].sta, sta);
          strcpy(CLEAR[i].pk[k].phase, phase);
          CLEAR[i].pk[k].abs_pk = abs_pk;
          CLEAR[i].pk[k].pk = pk;
          CLEAR[i].pk[k].amp = amp;
          CLEAR[i].pk[k].res = res;
          CLEAR[i].pk[k].baz = baz;
          CLEAR[i].pk[k].weig = weig;
          CLEAR[i].pk[k].mag = mag;
        }
      }
    }
  }

  // exclude the case that one pick is associated more than once
  for (i = 0; i < NN; i++) {
    for (j = 0; j < CLEAR[i].pscount; j++) {
      for (l = 0; l < NN, l != i; l++) {
        for (m = 0; m < CLEAR[l].pscount; m++) {
          if (memcmp(CLEAR[i].pk[j].net, CLEAR[l].pk[m].net, 5) == 0 &&
              memcmp(CLEAR[i].pk[j].sta, CLEAR[l].pk[m].sta, 8) == 0 &&
              memcmp(CLEAR[i].pk[j].phase, CLEAR[l].pk[m].phase, 10) == 0 &&
              fabs(CLEAR[i].pk[j].abs_pk - CLEAR[l].pk[m].abs_pk) < 1.0e-5) {
            // original one
            // if(fabs(CLEAR[i].pk[j].res) > fabs(CLEAR[l].pk[m].res)){
            // to eliminate large event splitting, suggested by Yen Joe Tan
            if (CLEAR[i].pscount < CLEAR[l].pscount ||
                (CLEAR[i].pscount == CLEAR[l].pscount &&
                 fabs(CLEAR[i].pk[j].res) > fabs(CLEAR[l].pk[m].res))) {
              CLEAR[i].pscount = CLEAR[i].pscount - 1;
              for (idx = j; idx < CLEAR[i].pscount; idx++)
                CLEAR[i].pk[idx] = CLEAR[i].pk[idx + 1];
            } else {
              CLEAR[l].pscount = CLEAR[l].pscount - 1;
              for (idx = m; idx < CLEAR[l].pscount; idx++)
                CLEAR[l].pk[idx] = CLEAR[l].pk[idx + 1];
            }
          }
        }
      }
    }
  }

  for (i = 0; i < NN; i++) {
    pcount = 0;
    scount = 0;
    psboth = 0;
    mag0 = (float *)malloc(CLEAR[i].pscount * sizeof(float));
    res0 = (float *)malloc(CLEAR[i].pscount * sizeof(float));
    for (j = 0; j < CLEAR[i].pscount; j++) {
      mag0[j] = CLEAR[i].pk[j].mag;
      res0[j] = CLEAR[i].pk[j].res;
      if (strcmp(CLEAR[i].pk[j].phase, "P") < 1.0e-5) {
        pcount++;
      } else if (strcmp(CLEAR[i].pk[j].phase, "S") < 1.0e-5) {
        scount++;
      }
      for (k = j + 1; k < CLEAR[i].pscount; k++) {
        if (memcmp(CLEAR[i].pk[j].net, CLEAR[i].pk[k].net, 5) == 0 &&
            memcmp(CLEAR[i].pk[j].sta, CLEAR[i].pk[k].sta, 8) == 0 &&
            memcmp(CLEAR[i].pk[j].phase, CLEAR[i].pk[k].phase, 10) != 0) {
          psboth++;
          break;
        }
      }
    }
    CLEAR[i].mag_median = CalculateMedian(mag0, CLEAR[i].pscount);
    CLEAR[i].mag_std =
        CalculateStd(mag0, CLEAR[i].mag_median, CLEAR[i].pscount);
    res_median = CalculateMedian(res0, CLEAR[i].pscount);
    CLEAR[i].std = CalculateStd(res0, res_median, CLEAR[i].pscount);
    CLEAR[i].pcount = pcount;
    CLEAR[i].scount = scount;
    CLEAR[i].pscount = pcount + scount;
    CLEAR[i].psboth = psboth;

    free(mag0);
    free(res0);
  }

  // select based on station azimuth gap
  for (i = 0; i < NN; i++) {
    gap0 = -100;
    for (j = 0; j < CLEAR[i].pscount - 1; j++) {
      k = j + 1;
      gap = CLEAR[i].pk[k].baz - CLEAR[i].pk[j].baz;
      if (gap > gap0)
        gap0 = gap;
    }
    // first and last azimuth
    k = CLEAR[i].pscount - 1;
    gap = 360 + CLEAR[i].pk[0].baz - CLEAR[i].pk[k].baz;
    if (gap > gap0) {
      gap0 = gap;
    }
    CLEAR[i].gap = gap0;
  }
}

// select one event within a short time window
int ReselectFinal(SELECT *RELC, int m) {
  int i, k, nps;
  char b[50];
  float a, c, d, e, f, g, h, o, p, q;
  extern int np0, ns0, nps0, npsboth0;

  for (i = 0; i < m; i++) {
    for (k = (i + 1); k < m; k++) {
      if (RELC[i].atime1 > RELC[k].atime1) {
        a = RELC[i].num1;
        strcpy(b, RELC[i].otime1);
        c = RELC[i].atime1;
        d = RELC[i].std1;
        e = RELC[i].lat1;
        f = RELC[i].lon1;
        g = RELC[i].dep1;
        h = RELC[i].nofp1;
        o = RELC[i].nofs1;
        p = RELC[i].ntotal1;
        q = RELC[i].nofps1;

        RELC[i].num1 = RELC[k].num1;
        strcpy(RELC[i].otime1, RELC[k].otime1);
        RELC[i].atime1 = RELC[k].atime1;
        RELC[i].std1 = RELC[k].std1;
        RELC[i].lat1 = RELC[k].lat1;
        RELC[i].lon1 = RELC[k].lon1;
        RELC[i].dep1 = RELC[k].dep1;
        RELC[i].nofp1 = RELC[k].nofp1;
        RELC[i].nofs1 = RELC[k].nofs1;
        RELC[i].ntotal1 = RELC[k].ntotal1;
        RELC[i].nofps1 = RELC[k].nofps1;

        RELC[k].num1 = a;
        strcpy(RELC[k].otime1, b);
        RELC[k].atime1 = c;
        RELC[k].std1 = d;
        RELC[k].lat1 = e;
        RELC[k].lon1 = f;
        RELC[k].dep1 = g;
        RELC[k].nofp1 = h;
        RELC[k].nofs1 = o;
        RELC[k].ntotal1 = p;
        RELC[k].nofps1 = q;
      }
    }
  }

  // exclude the case – one event is associated twice
  for (i = 1; i < m; i++) {
    for (k = 0; k < m, k != i; k++) {
      if (fabs(RELC[i].atime1 - RELC[k].atime1) < 1.0 * tint) {
        if (RELC[i].ntotal1 > RELC[k].ntotal1 ||
            (RELC[i].ntotal1 == RELC[k].ntotal1 &&
             RELC[i].std1 < RELC[k].std1)) {
          RELC[k].atime1 = 1.0e8;
        } else {
          RELC[i].atime1 = 1.0e8;
        }
      }
    }
  }

  for (i = 0; i < m; i++) {
    if (RELC[i].nofp1 < np0 || RELC[i].nofs1 < ns0 || RELC[i].ntotal1 < nps0 ||
        RELC[i].nofps1 < npsboth0)
      RELC[i].atime1 = 1.0e8;
  }

  for (i = 0; i < m; i++) {
    for (k = (i + 1); k < m; k++) {
      if (RELC[i].atime1 > RELC[k].atime1) {
        a = RELC[i].num1;
        strcpy(b, RELC[i].otime1);
        c = RELC[i].atime1;
        d = RELC[i].std1;
        e = RELC[i].lat1;
        f = RELC[i].lon1;
        g = RELC[i].dep1;
        h = RELC[i].nofp1;
        o = RELC[i].nofs1;
        p = RELC[i].ntotal1;
        q = RELC[i].nofps1;

        RELC[i].num1 = RELC[k].num1;
        strcpy(RELC[i].otime1, RELC[k].otime1);
        RELC[i].atime1 = RELC[k].atime1;
        RELC[i].std1 = RELC[k].std1;
        RELC[i].lat1 = RELC[k].lat1;
        RELC[i].lon1 = RELC[k].lon1;
        RELC[i].dep1 = RELC[k].dep1;
        RELC[i].nofp1 = RELC[k].nofp1;
        RELC[i].nofs1 = RELC[k].nofs1;
        RELC[i].ntotal1 = RELC[k].ntotal1;
        RELC[i].nofps1 = RELC[k].nofps1;

        RELC[k].num1 = a;
        strcpy(RELC[k].otime1, b);
        RELC[k].atime1 = c;
        RELC[k].std1 = d;
        RELC[k].lat1 = e;
        RELC[k].lon1 = f;
        RELC[k].dep1 = g;
        RELC[k].nofp1 = h;
        RELC[k].nofs1 = o;
        RELC[k].ntotal1 = p;
        RELC[k].nofps1 = q;
      }
    }
  }

  nps = m;
  for (i = 0; i < m; i++) {
    if (fabs(RELC[i].atime1 - 1.0e8) < 1 && RELC[i - 1].atime1 < MAXTIME) {
      nps = i;
      break;
    }
  }
  return nps;
}

float CalculateStd(float *arrValue, float median, int max) {
  int i;
  float std, temp;

  temp = 0.0;
  for (i = 0; i < max; i++) {
    temp += (arrValue[i] - median) * (arrValue[i] - median);
  }

  std = sqrt(temp / (max - 1));
  return std;
}

float CalculateMean(float *arrValue, int max) {
  float mean = 0.0;
  int i;
  for (i = 0; i < max; i++)
    mean = mean + arrValue[i];
  return mean / max;
}

float CalculateMedian(float *arrValue, int max) {
  float median = 0;
  float *value;
  int i, j;
  float temp;
  value = (float *)malloc(max * sizeof(float));
  for (i = 0; i < max; i++)
    value[i] = arrValue[i];

  for (i = 0; i < max; i++) {
    for (j = 0; j < max - i - 1; j++) {
      if (value[j] > value[j + 1]) {
        temp = value[j];
        value[j] = value[j + 1];
        value[j + 1] = temp;
      }
    }
  }
  if ((max % 2) == 1) {
    median = value[(max + 1) / 2 - 1];
  } else {
    median = (value[max / 2] + value[max / 2 - 1]) / 2;
  }
  free(value);
  return median;
}

int Readttime(char *name, TTT *TB, int nmax) {
  int i, test;
  FILE *infile;

  test = 0;

  while ((infile = fopen(name, "r")) == NULL) {
    fprintf(stdout, "Can not open file in ReadFile %s\n", name);
    exit(-1);
  }

  for (i = 0; i <= nmax; i++) {
    if (fscanf(infile, "%f %f %f %f %f %f %f %f %s %s\n", &TB[i].gdist,
               &TB[i].dep, &TB[i].ptime, &TB[i].stime, &TB[i].prayp,
               &TB[i].srayp, &TB[i].phslow, &TB[i].shslow, TB[i].pphase,
               TB[i].sphase) == EOF)
      test = 1;
    if (test == 1)
      break;
  }
  fclose(infile);
  return i;
}

int Readstation(char *name, STATION *ST, int nmax) {
  int i, test;
  FILE *infile;

  test = 0;

  while ((infile = fopen(name, "r")) == NULL) {
    fprintf(stdout, "Can not open file in ReadFile %s\n", name);
    exit(-1);
  }

  for (i = 0; i <= nmax; i++) {
    if (fscanf(infile, "%f %f %s %s %s %f\n", &ST[i].stlo, &ST[i].stla,
               ST[i].net, ST[i].sta, ST[i].comp, &ST[i].elev) == EOF)
      test = 1;
    if (test == 1)
      break;
  }
  fclose(infile);
  return i;
}

float Find_min(float **array, int n1, int n2) {
  int i, j;
  float amin;

  amin = 1.0e8;
  for (i = 0; i < n1; i++) {
    for (j = 0; j < n2; j++) {
      if (array[i][j] < amin) {
        amin = array[i][j];
      }
    }
  }
  return amin;
}

float Find_max(float **array, int n1, int n2) {
  int i, j;
  float amin;

  amin = -1.0e8;
  for (i = 0; i < n1; i++) {
    for (j = 0; j < n2; j++) {
      if (array[i][j] > amin && array[i][j] < 1.0e8) {
        amin = array[i][j];
      }
    }
  }
  return amin;
}

void Find_min_loc(float **array, int n1, int n2, float *amin, int *m,
                  int *n) {
  int i, j;

  *amin = 1.0e8;
  for (i = 0; i < n1; i++) {
    for (j = 0; j < n2; j++) {
      if (array[i][j] < *amin) {
        *amin = array[i][j];
        *m = i;
        *n = j;
      }
    }
  }
}

// find largest Nps with effective triggers
int DetermineNg(TRIG **ar1, TRIG **ar2, int n1, int n2) {
  int i, j, Nps1, Nps0;
  Nps1 = 0;
  Nps0 = 0;
  for (i = 0; i < n1; i++) {
    for (j = 1; j < n2; j++) {
      if (fabs(ar1[i][j].trig - 1.0e8) < 1 && ar1[i][j - 1].trig <= MAXTIME) {
        Nps0 = j;
        break;
      }
    }
    if (Nps0 > Nps1) {
      Nps1 = Nps0;
    }
  }

  for (i = 0; i < n1; i++) {
    for (j = 1; j < n2; j++) {
      if (fabs(ar2[i][j].trig - 1.0e8) < 1 && ar2[i][j - 1].trig <= MAXTIME) {
        Nps0 = j;
        break;
      }
    }
    if (Nps0 > Nps1) {
      Nps1 = Nps0;
    }
  }
  return Nps1 + 1;
}

// find largest Np with effective triggers
int DetermineNp(float **ar1, int n1, int n2) {
  int i, j, Nps1, Nps0;
  Nps1 = 0;
  Nps0 = 0;
  for (i = 0; i < n1; i++) {
    for (j = 1; j < n2; j++) {
      if (fabs(ar1[i][j] - 1.0e8) < 1 && ar1[i][j - 1] <= MAXTIME) {
        Nps0 = j;
        break;
      }
    }
    if (Nps0 >= Nps1) {
      Nps1 = Nps0;
    }
  }
  return Nps1 + 1;
}

// find Np range with effective time window
int DetermineNprange(float **ar1, float tpmax, int Nst, int Nps) {
  int i, j, Nps0, Nps00;
  Nps00 = 0;
  Nps0 = 0;

  // determine the upper bound for tpmax
  for (i = 0; i < Nst; i++) {
    for (j = 1; j < Nps; j++) {
      if (ar1[i][j] > tpmax && ar1[i][j - 1] < tpmax) {
        Nps0 = j;
        break;
      }
    }
    if (Nps0 >= Nps00) {
      Nps00 = Nps0;
    }
  }
  return Nps00 + 1;
}

void DetermineNps0range(float **ar1, float **ar2, float tpmin, float tpmax,
                        float tsmin, float tsmax, int Nst, int Nps) {
  int i, j;
  extern int *np0_start, *np0_end, *ns0_start, *ns0_end;

  // determine the lower bound for tpmin and upper bound for tpmax
  for (i = 0; i < Nst; i++) {
    np0_start[i] = 0;
    for (j = 1; j < Nps; j++) {
      if (ar1[i][j] > tpmin && ar1[i][j - 1] < tpmin) {
        np0_start[i] = j - 1;
        break;
      }
    }
  }
  for (i = 0; i < Nst; i++) {
    np0_end[i] = 0;
    for (j = 1; j < Nps; j++) {
      if (ar1[i][j] > tpmax && ar1[i][j - 1] < tpmax) {
        np0_end[i] = j;
        break;
      }
    }
  }
  // determine the lower bound for tsmin and upper bound for tsmax
  for (i = 0; i < Nst; i++) {
    ns0_start[i] = 0;
    for (j = 1; j < Nps; j++) {
      if (ar2[i][j] > tsmin && ar2[i][j - 1] < tsmin) {
        ns0_start[i] = j - 1;
        break;
      }
    }
  }
  for (i = 0; i < Nst; i++) {
    ns0_end[i] = 0;
    for (j = 1; j < Nps; j++) {
      if (ar2[i][j] > tsmax && ar2[i][j - 1] < tsmax) {
        ns0_end[i] = j;
        break;
      }
    }
  }
}

void SortTriggers0(TRIG **tgp, TRIG **tgs, float **array1, float **array2,
                   float **pamp, float **samp, float **pweight,
                   float **sweight, int m, int n) {
  int i, j, k, l;
  float a, b, c;

  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      for (k = (j + 1); k < n; ++k) {
        if (tgp[i][j].trig > tgp[i][k].trig) {
          a = tgp[i][j].trig;
          b = tgp[i][j].weight;
          c = tgp[i][j].amp;
          tgp[i][j].trig = tgp[i][k].trig;
          tgp[i][j].weight = tgp[i][k].weight;
          tgp[i][j].amp = tgp[i][k].amp;
          tgp[i][k].trig = a;
          tgp[i][k].weight = b;
          tgp[i][k].amp = c;
        }
        if (tgs[i][j].trig > tgs[i][k].trig) {
          a = tgs[i][j].trig;
          b = tgs[i][j].weight;
          c = tgs[i][j].amp;
          tgs[i][j].trig = tgs[i][k].trig;
          tgs[i][j].weight = tgs[i][k].weight;
          tgs[i][j].amp = tgs[i][k].amp;
          tgs[i][k].trig = a;
          tgs[i][k].weight = b;
          tgs[i][k].amp = c;
        }
      }
    }
  }

  for (i = 0; i < m; i++) {
    array1[i][0] = tgp[i][0].trig;
    array2[i][0] = tgs[i][0].trig;
    pamp[i][0] = tgp[i][0].amp;
    samp[i][0] = tgs[i][0].amp;
    pweight[i][0] = tgp[i][0].weight;
    sweight[i][0] = tgs[i][0].weight;
    for (j = 1; j < n; j++) {
      if (tgp[i][j].trig - tgp[i][j - 1].trig < nrt * ptw) {
        if (tgp[i][j].weight > tgp[i][j - 1].weight) {
          array1[i][j] = tgp[i][j].trig;
          pamp[i][j] = tgp[i][j].amp;
          pweight[i][j] = tgp[i][j].weight;
          array1[i][j - 1] = 1.0e8;
          pamp[i][j - 1] = 0.0;
          pweight[i][j - 1] = 0.0;
        } else {
          array1[i][j] = 1.0e8;
          pamp[i][j] = 0.0;
          pweight[i][j] = 0.0;
        }
      } else {
        array1[i][j] = tgp[i][j].trig;
        pamp[i][j] = tgp[i][j].amp;
        pweight[i][j] = tgp[i][j].weight;
      }

      if (tgs[i][j].trig - tgs[i][j - 1].trig < nrt * stw) {
        if (tgs[i][j].weight > tgs[i][j - 1].weight) {
          array2[i][j] = tgs[i][j].trig;
          samp[i][j] = tgs[i][j].amp;
          sweight[i][j] = tgs[i][j].weight;
          array2[i][j - 1] = 1.0e8;
          samp[i][j - 1] = 0.0;
          sweight[i][j - 1] = 0.0;
        } else {
          array2[i][j] = 1.0e8;
          samp[i][j] = 0.0;
          sweight[i][j] = 0.0;
        }
      } else {
        array2[i][j] = tgs[i][j].trig;
        samp[i][j] = tgs[i][j].amp;
        sweight[i][j] = tgs[i][j].weight;
      }
    }
  }

  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      for (k = (j + 1); k < n; ++k) {
        if (array1[i][j] > array1[i][k]) {
          a = array1[i][j];
          b = pamp[i][j];
          c = pweight[i][j];
          array1[i][j] = array1[i][k];
          pamp[i][j] = pamp[i][k];
          pweight[i][j] = pweight[i][k];
          array1[i][k] = a;
          pamp[i][k] = b;
          pweight[i][k] = c;
        }
        if (array2[i][j] > array2[i][k]) {
          a = array2[i][j];
          b = samp[i][j];
          c = sweight[i][j];
          array2[i][j] = array2[i][k];
          samp[i][j] = samp[i][k];
          sweight[i][j] = sweight[i][k];
          array2[i][k] = a;
          samp[i][k] = b;
          sweight[i][k] = c;
        }
      }
    }
  }
}

void DeleteOne(float **array, int Nst0, int Nps0, int Nloc) {
  int i;
  for (i = Nloc; i < Nps0 - 1; i++) {
    array[Nst0][i] = array[Nst0][i + 1];
  }
  array[Nst0][Nps0 - 1] = 1.0e8;
}

void Sortpscounts(float **pscounts0, int np) {
  int i, j, k;
  float a, b, c, d, e, f, g, h, p, q;

  for (i = 0; i < np; i++) {
    for (j = (i + 1); j < np; j++) {
      if (pscounts0[i][7] > pscounts0[j][7] ||
          (pscounts0[i][7] == pscounts0[j][7] &&
           pscounts0[i][6] < pscounts0[j][6])) {

        a = pscounts0[i][0];
        b = pscounts0[i][1];
        c = pscounts0[i][2];
        d = pscounts0[i][3];
        e = pscounts0[i][4];
        f = pscounts0[i][5];
        g = pscounts0[i][6];
        h = pscounts0[i][7];
        q = pscounts0[i][8];
        p = pscounts0[i][9];

        for (k = 0; k < 10; k++) {
          pscounts0[i][k] = pscounts0[j][k];
        }

        pscounts0[j][0] = a;
        pscounts0[j][1] = b;
        pscounts0[j][2] = c;
        pscounts0[j][3] = d;
        pscounts0[j][4] = e;
        pscounts0[j][5] = f;
        pscounts0[j][6] = g;
        pscounts0[j][7] = h;
        pscounts0[j][8] = q;
        pscounts0[j][9] = p;
      }
    }
  }
}

void Accounttriggers_homo(float lat0, float lon0, float dep, float latref,
                          float lonref, float elevref, int l) {
  int pcount, scount, ps;
  int i, j, k;
  float GCarc, baz, median, std, ptemp;
  float tp0_cal, tp_cal, ts_cal, tp_pre, ts_pre, tp_pre_b, tp_pre_e, ts_pre_b,
      ts_pre_e;
  extern float vp0, vs0, s_vp0, s_vs0;
  extern float nrt, ptw, stw, tpmin0;
  extern int np0, ns0, nps0, npsboth0, Nst, NNps;
  extern float **ptrig0, **strig0;
  extern int *np0_start, *np0_end, *ns0_start, *ns0_end;
  extern STATION *ST;
  extern float **pscounts;
  float *torg, *stagap, gap0, gaptemp, gap;
  extern float dtps;
  extern float GCarc0, std0;
  int puse, psboth;

  pcount = 0;
  scount = 0;
  ps = 0;

  torg = (float *)malloc(2 * Nst * sizeof(float));
  for (k = 0; k < 2 * Nst; k++)
    torg[k] = 0.0;
  stagap = (float *)malloc(2 * Nst * sizeof(float));
  for (k = 0; k < 2 * Nst; k++)
    stagap[k] = 0.0;

  ddistaz(lat0, lon0, latref, lonref, &GCarc, &baz);
  tp0_cal = sqrt((GCarc * 111.19) * (GCarc * 111.19) + dep * dep) / vp0 +
            elevref / s_vp0;

  psboth = 0;
  for (i = 0; i < Nst; i++) {
    ddistaz(ST[i].stla, ST[i].stlo, lat0, lon0, &GCarc, &baz);
    tp_cal = sqrt((GCarc * 111.19) * (GCarc * 111.19) + dep * dep) / vp0 +
             ST[i].elev / s_vp0;
    ts_cal = sqrt((GCarc * 111.19) * (GCarc * 111.19) + dep * dep) / vs0 +
             ST[i].elev / s_vs0;

    tp_pre = tpmin0 - tp0_cal + tp_cal;
    ts_pre = tpmin0 - tp0_cal + ts_cal;

    tp_pre_b = tp_pre - nrt * ptw / 2.0;
    tp_pre_e = tp_pre + nrt * ptw / 2.0;
    ts_pre_b = ts_pre - nrt * stw / 2.0;
    ts_pre_e = ts_pre + nrt * stw / 2.0;

    if (tp_pre_b < 0.0)
      tp_pre_b = 0.0;
    if (ts_pre_b < 0.0)
      ts_pre_b = 0.0;
    if (tp_pre_e > MAXTIME)
      tp_pre_e = MAXTIME;
    if (ts_pre_e > MAXTIME)
      ts_pre_e = MAXTIME;

    ptemp = -100;
    puse = 0;
    for (j = np0_start[i]; j < np0_end[i]; j++) {
      if (ptrig0[i][j] > tp_pre_b && ptrig0[i][j] < tp_pre_e &&
          GCarc < GCarc0) {
        torg[ps] = ptrig0[i][j] - tp_cal;
        stagap[ps] = baz;
        pcount = pcount + 1;
        ps = ps + 1;
        puse = 1;
        ptemp = ptrig0[i][j];
        break;
      }
    }

    // dtps: to remove some false S picks (they may be P picks but wrongly
    // identified as S picks, it happens!)
    for (j = ns0_start[i]; j < ns0_end[i]; j++) {
      if ((ts_pre - tp_pre) > dtps && fabs(ptemp - strig0[i][j]) > dtps &&
          strig0[i][j] > ts_pre_b && strig0[i][j] < ts_pre_e &&
          GCarc < GCarc0) {
        torg[ps] = strig0[i][j] - ts_cal;
        stagap[ps] = baz;
        scount = scount + 1;
        ps = ps + 1;
        if (puse == 1) {
          psboth++;
        }
        break;
      }
    }
  }

  if (pcount >= np0 && scount >= ns0 && ps >= nps0 && psboth >= npsboth0) {
    for (i = 0; i < ps; i++) {
      for (j = i; j < ps; j++) {
        if (stagap[j] < stagap[i]) {
          gaptemp = stagap[i];
          stagap[i] = stagap[j];
          stagap[j] = gaptemp;
        }
      }
    }

    gap0 = -100;
    for (i = 0; i < ps - 1; i++) {
      j = i + 1;
      gap = stagap[j] - stagap[i];
      if (gap > gap0)
        gap0 = gap;
    }
    gap = 360 + stagap[0] - stagap[ps - 1];
    if (gap > gap0)
      gap0 = gap;

    // median = CalculateMean(torg,ps);
    median = CalculateMedian(torg, ps);
    // median = (int)(median*1000.0+0.5)/1000.0;
    std = CalculateStd(torg, median, ps);
    pscounts[l][0] = lat0;
    pscounts[l][1] = lon0;
    pscounts[l][2] = dep;
    pscounts[l][3] = median;
    pscounts[l][4] = pcount;
    pscounts[l][5] = scount;
    pscounts[l][6] = std;
    pscounts[l][7] = ps;
    pscounts[l][8] = gap0;
    pscounts[l][9] = psboth;
  } else {
    pscounts[l][0] = lat0;
    pscounts[l][1] = lon0;
    pscounts[l][2] = dep;
    pscounts[l][3] = -1.0e8;
    pscounts[l][4] = pcount;
    pscounts[l][5] = scount;
    pscounts[l][6] = 1.0e8;
    pscounts[l][7] = ps;
    pscounts[l][8] = 1.0e8;
    pscounts[l][9] = psboth;
  }
  free(torg);
  free(stagap);
}

void Accounttriggers_layer(float lat0, float lon0, float dep, float latref,
                           float lonref, float elevref, int l) {
  int pcount, scount, ps;
  int i, j, k, ig, ih;
  float GCarc, baz, median, std, ptemp;
  float tp0_cal, tp_cal, ts_cal, tp_pre, ts_pre, tp_pre_b, tp_pre_e, ts_pre_b,
      ts_pre_e;
  extern float vp0, vs0, s_vp0, s_vs0;
  extern float nrt, ptw, stw, tpmin0;
  extern int np0, ns0, nps0, npsboth0, Nst, NNps;
  extern float **ptrig0, **strig0;
  extern int *np0_start, *np0_end, *ns0_start, *ns0_end;
  extern STATION *ST;
  extern float **pscounts;
  extern float trx, tdx, tdh, dtps;
  extern float GCarc0, std0;
  float *torg, *stagap, gap0, gaptemp, gap;
  int puse, psboth;

  pcount = 0;
  scount = 0;
  ps = 0;

  torg = (float *)malloc(2 * Nst * sizeof(float));
  for (k = 0; k < 2 * Nst; k++)
    torg[k] = 0.0;
  stagap = (float *)malloc(2 * Nst * sizeof(float));
  for (k = 0; k < 2 * Nst; k++)
    stagap[k] = 0.0;

  ddistaz(lat0, lon0, latref, lonref, &GCarc, &baz);
  ih = round(dep / tdh);
  ig = ih * rint(trx / tdx) + rint(GCarc / tdx);
  tp0_cal = TB[ig].ptime + (GCarc - TB[ig].gdist) * TB[ig].prayp +
            (dep - TB[ig].dep) * TB[ig].phslow + elevref / s_vp0;

  psboth = 0;
  for (i = 0; i < Nst; i++) {
    ddistaz(ST[i].stla, ST[i].stlo, lat0, lon0, &GCarc, &baz);
    ih = rint(dep / tdh);
    ig = ih * rint(trx / tdx) + rint(GCarc / tdx);
    tp_cal = TB[ig].ptime + (GCarc - TB[ig].gdist) * TB[ig].prayp +
             (dep - TB[ig].dep) * TB[ig].phslow + ST[i].elev / s_vp0;
    ts_cal = TB[ig].stime + (GCarc - TB[ig].gdist) * TB[ig].srayp +
             (dep - TB[ig].dep) * TB[ig].shslow + ST[i].elev / s_vs0;

    tp_pre = tpmin0 - tp0_cal + tp_cal;
    ts_pre = tpmin0 - tp0_cal + ts_cal;

    tp_pre_b = tp_pre - nrt * ptw / 2.0;
    tp_pre_e = tp_pre + nrt * ptw / 2.0;
    ts_pre_b = ts_pre - nrt * stw / 2.0;
    ts_pre_e = ts_pre + nrt * stw / 2.0;

    if (tp_pre_b < 0.0)
      tp_pre_b = 0.0;
    if (ts_pre_b < 0.0)
      ts_pre_b = 0.0;
    if (tp_pre_e > MAXTIME)
      tp_pre_e = MAXTIME;
    if (ts_pre_e > MAXTIME)
      ts_pre_e = MAXTIME;

    ptemp = -100;
    puse = 0;
    for (j = np0_start[i]; j < np0_end[i]; j++) {
      if (ptrig0[i][j] > tp_pre_b && ptrig0[i][j] < tp_pre_e &&
          GCarc < GCarc0) {
        torg[ps] = ptrig0[i][j] - tp_cal;
        stagap[ps] = baz;
        pcount = pcount + 1;
        ps = ps + 1;
        puse = 1;
        ptemp = ptrig0[i][j];
        break;
      }
    }

    // dtps: to remove some false S picks (they may be P picks but wrongly
    // identified as S picks, it happens!)
    for (j = ns0_start[i]; j < ns0_end[i]; j++) {
      if ((ts_pre - tp_pre) > dtps && fabs(ptemp - strig0[i][j]) > dtps &&
          strig0[i][j] > ts_pre_b && strig0[i][j] < ts_pre_e &&
          GCarc < GCarc0) {
        torg[ps] = strig0[i][j] - ts_cal;
        stagap[ps] = baz;
        scount = scount + 1;
        ps = ps + 1;
        if (puse == 1) {
          psboth++;
        }
        break;
      }
    }
  }

  if (pcount >= np0 && scount >= ns0 && ps >= nps0 && psboth >= npsboth0) {
    for (i = 0; i < ps; i++) {
      for (j = i; j < ps; j++) {
        if (stagap[j] < stagap[i]) {
          gaptemp = stagap[i];
          stagap[i] = stagap[j];
          stagap[j] = gaptemp;
        }
      }
    }

    gap0 = -100;
    for (i = 0; i < ps - 1; i++) {
      j = i + 1;
      gap = stagap[j] - stagap[i];
      if (gap > gap0)
        gap0 = gap;
    }
    gap = 360 + stagap[0] - stagap[ps - 1];
    if (gap > gap0)
      gap0 = gap;

    // median = CalculateMean(torg,ps);
    median = CalculateMedian(torg, ps);
    // median = (int)(median*1000.0+0.5)/1000.0;
    std = CalculateStd(torg, median, ps);
    pscounts[l][0] = lat0;
    pscounts[l][1] = lon0;
    pscounts[l][2] = dep;
    pscounts[l][3] = median;
    pscounts[l][4] = pcount;
    pscounts[l][5] = scount;
    pscounts[l][6] = std;
    pscounts[l][7] = ps;
    pscounts[l][8] = gap0;
    pscounts[l][9] = psboth;
  } else {
    pscounts[l][0] = lat0;
    pscounts[l][1] = lon0;
    pscounts[l][2] = dep;
    pscounts[l][3] = -1.0e8;
    pscounts[l][4] = pcount;
    pscounts[l][5] = scount;
    pscounts[l][6] = 1.0e8;
    pscounts[l][7] = ps;
    pscounts[l][8] = 1.0e8;
    pscounts[l][9] = psboth;
  }
  free(torg);
  free(stagap);
}

/*
* Modified by M. Zhang
c Subroutine to calculate the Great Circle Arc distance
c    between two sets of geographic coordinates
c
c Given:  stalat => Latitude of first point (+N, -S) in degrees
c         stalon => Longitude of first point (+E, -W) in degrees
c         evtlat => Latitude of second point
c         evtlon => Longitude of second point
c
c Returns:  delta => Great Circle Arc distance in degrees
c           az    => Azimuth from pt. 1 to pt. 2 in degrees
c           baz   => Back Azimuth from pt. 2 to pt. 1 in degrees
c
c If you are calculating station-epicenter pairs, pt. 1 is the station
c
c Equations take from Bullen, pages 154, 155
c
c T. Owens, September 19, 1991
c           Sept. 25 -- fixed az and baz calculations
c
P. Crotwell, Setember 27, 1994
    Converted to c to fix annoying problem of fortran giving wrong
       answers if the input doesn't contain a decimal point.
*/
void ddistaz(float stalat, float stalon, float evtlat, float evtlon,
             float *delta, float *baz) {
  // float stalat, stalon, evtlat, evtlon;
  // float delta, az, baz;
  float scolat, slon, ecolat, elon;
  float a, b, c, d, e, aa, bb, cc, dd, ee, g, gg, h, hh, k, kk;
  float rhs1, rhs2, sph, rad, del, daz, dbaz, pi, piby2;
  /*
  stalat = atof(argv[1]);
  stalon = atof(argv[2]);
  evtlat = atof(argv[3]);
  evtlon = atof(argv[4]);
  */
  pi = 3.141592654;
  piby2 = pi / 2.0;
  rad = 2. * pi / 360.0;
  sph = 1.0 / 298.257;

  scolat = piby2 - atan((1. - sph) * (1. - sph) * tan(stalat * rad));
  ecolat = piby2 - atan((1. - sph) * (1. - sph) * tan(evtlat * rad));
  slon = stalon * rad;
  elon = evtlon * rad;
  a = sin(scolat) * cos(slon);
  b = sin(scolat) * sin(slon);
  c = cos(scolat);
  d = sin(slon);
  e = -cos(slon);
  g = -c * e;
  h = c * d;
  k = -sin(scolat);
  aa = sin(ecolat) * cos(elon);
  bb = sin(ecolat) * sin(elon);
  cc = cos(ecolat);
  dd = sin(elon);
  ee = -cos(elon);
  gg = -cc * ee;
  hh = cc * dd;
  kk = -sin(ecolat);
  del = acos(a * aa + b * bb + c * cc);
  *delta = del / rad; // delta

  rhs1 = (aa - d) * (aa - d) + (bb - e) * (bb - e) + cc * cc - 2.;
  rhs2 = (aa - g) * (aa - g) + (bb - h) * (bb - h) + (cc - k) * (cc - k) - 2.;
  dbaz = atan2(rhs1, rhs2);
  if (dbaz < 0.0) {
    dbaz = dbaz + 2 * pi;
  }
  *baz = dbaz / rad; // baz
  if (fabs(*baz - 360.) < .00001)
    *baz = 0.0;
}
