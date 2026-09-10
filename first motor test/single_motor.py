# Single Motor example using the LEGO Education Python API
# Library: https://github.com/LEGO/LEGOEducation
# Install with: pip install legoeducation

import legoeducation as le

# Update these to match your motor's Connection Card (color + serial number
# printed on the card). See connect.md in the LEGOEducation repo for the
# full list of card colors.
# There is no LEGO_COLOR_PINK constant in this library. The closest
# Connection Card color to pink is MAGENTA.
CARD_COLOR = le.LEGO_COLOR_MAGENTA
CARD_SERIAL = '0998'

# 1. Create the motor object
motor = le.SingleMotor()

# 2. Make sure the motor is charged, powered on, and Bluetooth-broadcasting,
#    then connect to it
motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)

# 3. Check the connection succeeded
if not motor.connected:
    print('Error connecting to Single Motor.')
    exit(1)

print('Connected to Single Motor.')

# 4. Run the motor
motor.motor_run_for_degrees(
    degrees=360,
    direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
    speed=50,
)

print(f'Final position: {motor.motor.position}')

# 5. Disconnect when done
motor.disconnect()
exit(0)
