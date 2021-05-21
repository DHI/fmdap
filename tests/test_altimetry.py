import re
import sys, os
import pytest
from datetime import datetime
from fmdap import altimetry


user = os.environ["DHI_ALTIMETRY_USER"]
pwd = os.environ["DHI_ALTIMETRY_PASSWORD"]


def test_create_altimetry_query():
    alti = altimetry.AltimetryDHI()
    q = alti.create_altimetry_query()
    expected = "https://altimetry-api.dhigroup.com/query-csv?bbox=-11.913345,48.592117,12.411167,63.084148&satellites=3a,3b&qual_filters=dhi_combined&numeric=false"
    assert q == expected


def help_create_query_url(alti):
    __tracebackhide__ = True

    area = "lon=10.9&lat=55.9&radius=10"
    start_date = "20180115"
    end_date = "20180201"
    qa = ""
    satellites = ["3a"]

    url_query = alti.create_altimetry_query(
        area=area,
        start_date=start_date,
        end_date=end_date,
        quality_filter=qa,
        satellites=satellites,
    )
    return url_query


def test_get_altimetry_from_api():
    alti = altimetry.AltimetryDHI(user, pwd)
    url_query = help_create_query_url(alti)
    df = alti.get_altimetry_from_api(url_query)

    assert df["lat"].values[5] == pytest.approx(55.84644499, 1e-4)
    assert df["adt"].values[4] == pytest.approx(0.2269, 1e-3)
    assert df.shape == (6, 11)


# def test_get_altimetry_from_api_fail():
#     fake_user = 'IAmaFakeKey'
#     alti = altimetry.AltimetryDHI(fake_user)
#     url_query = help_create_query_url(alti)
#     with pytest.raises(altimetry.APIAuthenticationFailed):
#         df = alti.get_altimetry_from_api(url_query, fake_user)


# def test_get_altimetry_from_api():
#     alti = altimetry.AltimetryDHI(api_key)
#     area = 'lon=10.9&lat=55.9&radius=10000'
#     start_date = '2018-01-01'
#     end_date = '2018-02-01'
#     qa = ''
#     satellites = '3a',

#     url_query = alti.create_altimetry_query(area=area,
#                                     start_date=start_date, end_date=end_date,
#                                     quality_filter=qa, satellites=satellites)
#     df = alti.get_altimetry_from_api_old(url_query)

# assert df['lat'].values[8] == pytest.approx(55.846445, 1e-4)
# assert df['adt'].values[0] == pytest.approx(0.2262, 1e-3)
# assert df.shape == (9,11)


def test_parse_satellite_list():
    alti = altimetry.AltimetryDHI()
    sats = alti.parse_satellite_list("sentinels")
    sats = re.split("=|,", sats)
    assert sats[1] == "3a"
    assert sats[2] == "3b"


def test_get_data():
    alti = altimetry.AltimetryDHI(user, pwd)
    df = alti.get_data(
        area="bbox=1,55,2,57", end_date=datetime(2018, 2, 1), satellites="sentinels"
    )
    assert df.shape == (96, 11)
    assert df.lon[1] == 1.061956
    assert df.range_rms[-1] == 0.0571999996900558


def test_save_to_dfs0():
    alti = altimetry.AltimetryDHI(user, pwd)
    df = alti.get_data(
        area="bbox=1,55,2,57", start_date="20190401", end_date="20190601"
    )
    dfsfile = "test.dfs0"
    alti.save_to_dfs0(dfsfile, df)
    assert os.path.isfile(dfsfile)

    # clean up
    os.remove(dfsfile)
