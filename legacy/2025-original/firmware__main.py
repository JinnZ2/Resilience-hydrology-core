---

### `/firmware/esp32_basic/main.py`

```python
"""
Resilience Node - Basic Data Logger

Minimal firmware for ESP32 that logs temperature gradients.
Perfect for getting started - just logs data, no actuators needed.

Hardware:
- ESP32 dev board
- 2× DS18B20 temperature sensors
- MicroSD card (optional, for local storage)

Wiring:
- GPIO4 → DS18B20 #1 (ground sensor)
- GPIO5 → DS18B20 #2 (air sensor)
- Both sensors: 3.3V and GND
- 4.7kΩ pullup resistor on data lines
"""

import machine
import onewire
import ds18x20
import time
from machine import Pin, SoftSPI, SDCard
import os

class TemperatureLogger:
    def __init__(self, pin_ground=4, pin_air=5):
        """Initialize temperature sensors."""
        # Setup OneWire buses
        self.ow_ground = onewire.OneWire(Pin(pin_ground))
        self.ow_air = onewire.OneWire(Pin(pin_air))
        
        # Initialize DS18B20 sensors
        self.sensor_ground = ds18x20.DS18X20(self.ow_ground)
        self.sensor_air = ds18x20.DS18X20(self.ow_air)
        
        # Scan for devices
        self.addr_ground = self.sensor_ground.scan()
        self.addr_air = self.sensor_air.scan()
        
        if not self.addr_ground:
            print("Warning: No ground sensor found on GPIO4")
        if not self.addr_air:
            print("Warning: No air sensor found on GPIO5")
        
        # LED for status
        self.led = Pin(2, Pin.OUT)
        
        print("Temperature logger initialized")
    
    def read_temperatures(self):
        """Read both temperature sensors."""
        readings = {
            'timestamp': time.time(),
            'temp_ground': None,
            'temp_air': None,
            'delta_t': None
        }
        
        try:
            # Start conversion
            self.sensor_ground.convert_temp()
            self.sensor_air.convert_temp()
            time.sleep_ms(750)  # Wait for conversion
            
            # Read temperatures
            if self.addr_ground:
                readings['temp_ground'] = self.sensor_ground.read_temp(self.addr_ground[0])
            
            if self.addr_air:
                readings['temp_air'] = self.sensor_air.read_temp(self.addr_air[0])
            
            # Calculate gradient
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
        # Initialize SPI for SD card
        spi = SoftSPI(sck=Pin(18), mosi=Pin(23), miso=Pin(19))
        sd = SDCard(spi, Pin(5))
        
        # Mount SD card
        os.mount(sd, '/sd')
        print("SD card mounted at /sd")
        return True
    except Exception as e:
        print(f"Could not mount SD card: {e}")
        print("Logging to flash memory instead")
        return False


def main():
    """Main logging loop."""
    print("="*40)
    print("Resilience Node - Temperature Logger")
    print("="*40)
    print()
    
    # Initialize hardware
    logger = TemperatureLogger()
    has_sd = setup_sd_card()
    
    # Setup log file
    log_path = '/sd/temp_log.csv' if has_sd else 'temp_log.csv'
    
    # Write header if new file
    try:
        with open(log_path, 'r') as f:
            pass
    except:
        with open(log_path, 'w') as f:
            f.write("timestamp,temp_ground_C,temp_air_C,delta_t_C\n")
    
    print(f"Logging to: {log_path}")
    print("Press Ctrl+C to stop")
    print()
    
    # Main loop
    log_interval = 300  # seconds (5 minutes)
    
    while True:
        try:
            # Read sensors
            readings = logger.read_temperatures()
            
            # Display on console
            if readings['temp_ground'] is not None:
                print(f"Ground: {readings['temp_ground']:.2f}°C  ", end='')
            if readings['temp_air'] is not None:
                print(f"Air: {readings['temp_air']:.2f}°C  ", end='')
            if readings['delta_t'] is not None:
                print(f"ΔT: {readings['delta_t']:.2f}°C")
            else:
                print()
            
            # Write to log file
            log_entry = logger.format_log_entry(readings)
            with open(log_path, 'a') as f:
                f.write(log_entry)
            
            # Blink to show activity
            logger.blink_status()
            
            # Sleep until next reading
            time.sleep(log_interval)
            
        except KeyboardInterrupt:
            print("\nLogging stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)  # Wait a bit before retrying


if __name__ == '__main__':
    main()
