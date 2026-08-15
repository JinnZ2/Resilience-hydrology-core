"""
Resilience Node - Validation Logger

Logs everything needed to check the dew model against reality, which the basic
logger cannot do. This is the firmware side of closing O1 in
docs/research-log.md.

WHY THIS EXISTS

firmware/esp32_basic logs a ground/air temperature gradient. That is not enough
to test anything: the model predicts a COLLECTOR SURFACE temperature and a
nightly VOLUME, driven mostly by HUMIDITY, and the basic build measures none of
those three.

This firmware adds them:

  air temperature      DS18B20   - already had it
  surface temperature  DS18B20   - bonded to the collector plate. The model
                                   predicts this directly, so comparing it is
                                   the cheapest test of whether the physics is
                                   right at all
  relative humidity    SHT31     - the largest single driver of whether dew
                                   forms, and the current build is blind to it
  collected volume     tipping bucket on an interrupt - turns "85 ml, then the
                                   notebook stops" into a nightly series

Together these cost about $23 and roughly four hours. See
docs/build-guide.md, "The measurement kit".

RECORD THESE TWO BY HAND, ONCE

    COLLECTOR_AREA_M2  and  COLLECTOR_TILT_DEG   (below)

They are constants, not sensors, and they are the two numbers whose absence
currently makes the field data uninterpretable. A tape measure and a protractor.
Do not leave them at their defaults.

Hardware:
- ESP32 dev board
- 2x DS18B20 (air, collector surface) on separate OneWire pins
- SHT31-D humidity/temperature sensor on I2C
- Tipping-bucket rain gauge (reed switch) on an interrupt pin
- 4.7k pullup on each OneWire data line

Wiring:
- GPIO4  -> DS18B20 air sensor
- GPIO5  -> DS18B20 collector surface sensor
- GPIO21 -> SHT31 SDA
- GPIO22 -> SHT31 SCL
- GPIO27 -> tipping bucket reed switch (other side to GND)
"""

import time

import machine
import onewire
import ds18x20
from machine import Pin, I2C


# --- Site constants: MEASURE THESE AND EDIT THEM ----------------------------
# Defaults are deliberately implausible so that unedited values are obvious in
# the log rather than silently passing as real.
COLLECTOR_AREA_M2 = -1.0     # tape measure. See docs/research-log.md O1
COLLECTOR_TILT_DEG = -1.0    # protractor.   See docs/research-log.md O9

# Tipping bucket volume per tip, in mL. Calibrate by pouring a known volume
# through the gauge and counting tips.
ML_PER_TIP = 5.0

LOG_INTERVAL_S = 300         # 5 minutes
LOG_FILE = 'dew_log.csv'


class TippingBucket:
    """Counts reed-switch closures with debounce, on an interrupt."""

    DEBOUNCE_MS = 120

    def __init__(self, pin=27):
        self.tips = 0
        self._last_ms = 0
        self._pin = Pin(pin, Pin.IN, Pin.PULL_UP)
        self._pin.irq(trigger=Pin.IRQ_FALLING, handler=self._on_tip)

    def _on_tip(self, pin):
        now = time.ticks_ms()
        if time.ticks_diff(now, self._last_ms) < self.DEBOUNCE_MS:
            return          # contact bounce, not a second tip
        self._last_ms = now
        self.tips += 1

    @property
    def volume_ml(self):
        return self.tips * ML_PER_TIP


class SHT31:
    """
    Minimal SHT31-D driver: single-shot, high repeatability, clock stretching
    disabled. Enough to read temperature and humidity; no heater, no status
    register.
    """

    def __init__(self, i2c, addr=0x44):
        self.i2c = i2c
        self.addr = addr

    def read(self):
        """Returns (temperature_c, relative_humidity_fraction) or (None, None)."""
        try:
            self.i2c.writeto(self.addr, b'\x24\x00')   # no clock stretch, high rep
            time.sleep_ms(20)
            data = self.i2c.readfrom(self.addr, 6)
        except OSError:
            return None, None
        if len(data) != 6:
            return None, None
        raw_t = (data[0] << 8) | data[1]
        raw_h = (data[3] << 8) | data[4]
        temp_c = -45.0 + 175.0 * raw_t / 65535.0
        humidity = raw_h / 65535.0
        return temp_c, humidity


class ValidationLogger:
    """Logs the four quantities the model needs to be checkable."""

    def __init__(self, pin_air=4, pin_surface=5, pin_bucket=27,
                 sda=21, scl=22):
        self.ow_air = onewire.OneWire(Pin(pin_air))
        self.ow_surface = onewire.OneWire(Pin(pin_surface))
        self.sensor_air = ds18x20.DS18X20(self.ow_air)
        self.sensor_surface = ds18x20.DS18X20(self.ow_surface)

        self.roms_air = self.sensor_air.scan()
        self.roms_surface = self.sensor_surface.scan()

        i2c = I2C(0, sda=Pin(sda), scl=Pin(scl), freq=100000)
        self.sht = SHT31(i2c)
        self.bucket = TippingBucket(pin_bucket)

    def read_temp(self, sensor, roms):
        if not roms:
            return None
        sensor.convert_temp()
        time.sleep_ms(750)          # DS18B20 12-bit conversion time
        try:
            return sensor.read_temp(roms[0])
        except Exception:
            return None

    def read_all(self):
        t_air = self.read_temp(self.sensor_air, self.roms_air)
        t_surface = self.read_temp(self.sensor_surface, self.roms_surface)
        t_sht, humidity = self.sht.read()
        if t_air is None:
            t_air = t_sht           # SHT31 as fallback air temperature
        return {
            'uptime_s': time.time(),
            't_air_c': t_air,
            't_surface_c': t_surface,
            'rh': humidity,
            'tips': self.bucket.tips,
            'volume_ml': self.bucket.volume_ml,
        }

    def write_header(self):
        try:
            with open(LOG_FILE, 'r'):
                return                      # already exists, keep appending
        except OSError:
            pass
        with open(LOG_FILE, 'w') as f:
            f.write('# area_m2={:.3f} tilt_deg={:.1f} ml_per_tip={:.2f}\n'.format(
                COLLECTOR_AREA_M2, COLLECTOR_TILT_DEG, ML_PER_TIP))
            f.write('uptime_s,t_air_c,t_surface_c,rh,tips,volume_ml\n')

    def log(self, reading):
        def fmt(value, digits=2):
            return '' if value is None else ('{:.' + str(digits) + 'f}').format(value)

        line = '{},{},{},{},{},{}\n'.format(
            reading['uptime_s'], fmt(reading['t_air_c']),
            fmt(reading['t_surface_c']), fmt(reading['rh'], 3),
            reading['tips'], fmt(reading['volume_ml'], 1))
        with open(LOG_FILE, 'a') as f:
            f.write(line)
        return line


def check_site_constants():
    """Refuse to pretend unmeasured constants are data."""
    problems = []
    if COLLECTOR_AREA_M2 <= 0:
        problems.append('COLLECTOR_AREA_M2 is unset - measure the collector')
    if COLLECTOR_TILT_DEG < 0:
        problems.append('COLLECTOR_TILT_DEG is unset - measure the angle')
    return problems


def main():
    print('Resilience Node - Validation Logger')

    for problem in check_site_constants():
        print('  WARNING: {}'.format(problem))
    if check_site_constants():
        print('  Logging will continue, but the data cannot be compared to any')
        print('  model until these are filled in. That is the whole point of')
        print('  this firmware. See docs/research-log.md, O1 and O9.')

    logger = ValidationLogger()
    logger.write_header()

    if not logger.roms_surface:
        print('  WARNING: no surface sensor found on GPIO5. The collector')
        print('  surface temperature is what the model actually predicts.')

    print('Logging every {}s to {}'.format(LOG_INTERVAL_S, LOG_FILE))
    print('uptime_s,t_air_c,t_surface_c,rh,tips,volume_ml')

    while True:
        reading = logger.read_all()
        print(logger.log(reading).strip())
        time.sleep(LOG_INTERVAL_S)


if __name__ == '__main__':
    main()
