"""
Resilience Node - Basic Data Logger

Minimal firmware for ESP32 that logs temperature gradients.
Just logs data, no actuators needed.

Hardware:
- ESP32 dev board
- 2x DS18B20 temperature sensors
- MicroSD card (optional, for local storage)

Wiring:
- GPIO4  -> DS18B20 #1 (ground sensor)
- GPIO5  -> DS18B20 #2 (air sensor)
- Both sensors: 3.3V and GND
- 4.7k ohm pullup resistor on data lines

Optional SD card (SoftSPI):
- GPIO18 -> SCK
- GPIO23 -> MOSI
- GPIO19 -> MISO
- GPIO15 -> CS   (was GPIO5, which collided with the air sensor -- see M-06)
"""

import machine
import onewire
import ds18x20
import time
from machine import Pin, SoftSPI, SDCard
import os


LOG_INTERVAL = 300  # seconds (5 minutes)

# SD card pins. CS must not collide with either OneWire pin (4, 5).
SD_SCK, SD_MOSI, SD_MISO, SD_CS = 18, 23, 19, 15


class TemperatureLogger:
    def __init__(self, pin_ground=4, pin_air=5):
        """Initialize temperature sensors."""
        if SD_CS in (pin_ground, pin_air):
            raise ValueError(
                f"SD_CS (GPIO{SD_CS}) collides with a sensor pin "
                f"(GPIO{pin_ground}, GPIO{pin_air})")

        self.ow_ground = onewire.OneWire(Pin(pin_ground))
        self.ow_air = onewire.OneWire(Pin(pin_air))

        self.sensor_ground = ds18x20.DS18X20(self.ow_ground)
        self.sensor_air = ds18x20.DS18X20(self.ow_air)

        self.addr_ground = self.sensor_ground.scan()
        self.addr_air = self.sensor_air.scan()

        if not self.addr_ground:
            print("Warning: No ground sensor found on GPIO4")
        if not self.addr_air:
            print("Warning: No air sensor found on GPIO5")

        self.led = Pin(2, Pin.OUT)
        print("Temperature logger initialized")

    def read_temperatures(self):
        """Read both temperature sensors."""
        readings = {
            'timestamp': time.time(),
            'temp_ground': None,
            'temp_air': None,
            'delta_t': None,
        }

        try:
            self.sensor_ground.convert_temp()
            self.sensor_air.convert_temp()
            time.sleep_ms(750)

            if self.addr_ground:
                readings['temp_ground'] = self.sensor_ground.read_temp(self.addr_ground[0])
            if self.addr_air:
                readings['temp_air'] = self.sensor_air.read_temp(self.addr_air[0])

            if readings['temp_ground'] is not None and readings['temp_air'] is not None:
                readings['delta_t'] = readings['temp_ground'] - readings['temp_air']
        except Exception as e:
            print(f"Error reading sensors: {e}")

        return readings

    def format_log_entry(self, readings):
        """Format readings as CSV line."""
        return (f"{readings['timestamp']},"
                f"{readings.get('temp_ground', 'NA')},"
                f"{readings.get('temp_air', 'NA')},"
                f"{readings.get('delta_t', 'NA')}\n")

    def blink_status(self):
        """Blink LED to show activity."""
        self.led.value(1)
        time.sleep_ms(100)
        self.led.value(0)


def setup_sd_card():
    """Initialize SD card for data logging."""
    try:
        spi = SoftSPI(sck=Pin(SD_SCK), mosi=Pin(SD_MOSI), miso=Pin(SD_MISO))
        sd = SDCard(spi, Pin(SD_CS))
        os.mount(sd, '/sd')
        print("SD card mounted at /sd")
        return True
    except Exception as e:
        print(f"Could not mount SD card: {e}")
        print("Logging to flash memory instead")
        return False


def main():
    """Main logging loop."""
    print("=" * 40)
    print("Resilience Node - Temperature Logger")
    print("=" * 40)
    print()

    logger = TemperatureLogger()
    has_sd = setup_sd_card()

    log_path = '/sd/temp_log.csv' if has_sd else 'temp_log.csv'

    try:
        with open(log_path, 'r'):
            pass
    except OSError:
        with open(log_path, 'w') as f:
            f.write("timestamp,temp_ground_C,temp_air_C,delta_t_C\n")

    print(f"Logging to: {log_path}")
    print("Press Ctrl+C to stop")
    print()

    while True:
        try:
            readings = logger.read_temperatures()

            if readings['temp_ground'] is not None:
                print(f"Ground: {readings['temp_ground']:.2f} C  ", end='')
            if readings['temp_air'] is not None:
                print(f"Air: {readings['temp_air']:.2f} C  ", end='')
            if readings['delta_t'] is not None:
                print(f"dT: {readings['delta_t']:.2f} C")
            else:
                print()

            log_entry = logger.format_log_entry(readings)
            with open(log_path, 'a') as f:
                f.write(log_entry)

            logger.blink_status()
            time.sleep(LOG_INTERVAL)

        except KeyboardInterrupt:
            print("\nLogging stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)


if __name__ == '__main__':
    main()
