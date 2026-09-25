#!/usr/bin/env python3
"""What would this miner earn? Live numbers from mempool.space.

Usage: tools/economics.py --mhs 128 --watts 30 [--usd-per-kwh 0.10]
                          [--rent-usd-per-hour 0.15] [--offline]

--mhs    measured hash rate of this machine in MH/s
--watts  power drawn while mining. This VM exposes no power counters, so
         pass an estimate. Cascade Lake Xeons draw ~5-10 W per busy core at
         package level, so ~20-40 W for 4 cores.
"""
import argparse
import json
import math
import urllib.request

API = 'https://mempool.space/api'
# Snapshot used with --offline or when the API is unreachable (2026-09-25).
SNAPSHOT = {'hashrate': 9.296e20, 'difficulty': 1.3276e14, 'usd': 83847.0,
            'fees_per_block': 372922336 / 144 / 1e8, 'height': 968566}
SUBSIDY = 3.125  # BTC per block until the 2028 halving (height 1,050,000)
# A current flagship ASIC, manufacturer spec (Bitmain Antminer S21 XP, 2024).
ASIC = {'name': 'Antminer S21 XP (spec)', 'ths': 270.0, 'watts': 3645.0}


def get(path):
    with urllib.request.urlopen(API + path, timeout=20) as r:
        return json.load(r)


def live():
    hr = get('/v1/mining/hashrate/3d')
    fees = get('/v1/mining/reward-stats/144')
    blocks = int(fees['endBlock']) - int(fees['startBlock']) + 1
    return {'hashrate': float(hr['currentHashrate']),
            'difficulty': float(hr['currentDifficulty']),
            'usd': float(get('/v1/prices')['USD']),
            'fees_per_block': int(fees['totalFee']) / blocks / 1e8,
            'height': int(get('/blocks/tip/height'))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mhs', type=float, required=True)
    ap.add_argument('--watts', type=float, required=True)
    ap.add_argument('--usd-per-kwh', type=float, default=0.10)
    ap.add_argument('--rent-usd-per-hour', type=float, default=0.15)
    ap.add_argument('--offline', action='store_true')
    a = ap.parse_args()

    src = 'snapshot 2026-09-25'
    net = SNAPSHOT
    if not a.offline:
        try:
            net = live()
            src = 'live mempool.space'
        except Exception as e:  # network down: fall back, but say so
            src = 'snapshot 2026-09-25 (live fetch failed: %s)' % e

    h = a.mhs * 1e6
    reward = SUBSIDY + net['fees_per_block']
    secs_per_block = net['difficulty'] * 2 ** 32 / h  # expected solo time
    blocks_per_day = 86400 / secs_per_block
    btc_day = blocks_per_day * reward
    usd_year = btc_day * 365 * net['usd']
    years = secs_per_block / (365.25 * 86400)
    p_year = 1 - math.exp(-1 / years)
    j_per_th = a.watts / (h / 1e12)
    power_usd_year = a.watts / 1000 * 24 * 365 * a.usd_per_kwh
    rent_usd_year = a.rent_usd_per_hour * 24 * 365
    asic_jth = ASIC['watts'] / ASIC['ths']
    asic_btc_day = 86400 / (net['difficulty'] * 2 ** 32 / (ASIC['ths'] * 1e12)) * reward

    print('Network (%s): height %d, hashrate %.3g H/s, difficulty %.4g, BTC $%.0f, '
          'reward %.4f BTC/block (subsidy %.3f + fees %.4f)\n'
          % (src, net['height'], net['hashrate'], net['difficulty'], net['usd'], reward,
             SUBSIDY, net['fees_per_block']))
    rows = [
        ('This miner', '%.1f MH/s' % a.mhs),
        ('Share of network hashrate', '%.2g' % (h / net['hashrate'])),
        ('Expected revenue', '%.3g BTC/day = $%.4f per year' % (btc_day, usd_year)),
        ('Expected time to find a block solo', '%.3g years' % years),
        ('Chance of a block within a year', '%.2g' % p_year),
        ('Energy efficiency (at %.0f W, estimated)' % a.watts, '%.3g J/TH' % j_per_th),
        ('%s' % ASIC['name'], '%.1f J/TH, %.0f TH/s' % (asic_jth, ASIC['ths'])),
        ('CPU energy per hash vs ASIC', '%.2gx worse' % (j_per_th / asic_jth)),
        ('Electricity at $%.2f/kWh' % a.usd_per_kwh, '$%.2f per year' % power_usd_year),
        ('Revenue / electricity cost', '%.2g' % (usd_year / power_usd_year)),
        ('Cloud VM rent at $%.2f/h' % a.rent_usd_per_hour, '$%.0f per year (%.2gx revenue)'
         % (rent_usd_year, rent_usd_year / usd_year)),
        ('ASIC revenue for comparison', '%.4f BTC/day = $%.0f per year per machine'
         % (asic_btc_day, asic_btc_day * 365 * net['usd'])),
    ]
    width = max(len(k) for k, _ in rows)
    print('| %s | value |\n|%s|---|' % ('quantity'.ljust(width), '-' * (width + 2)))
    for k, v in rows:
        print('| %s | %s |' % (k.ljust(width), v))


if __name__ == '__main__':
    main()
