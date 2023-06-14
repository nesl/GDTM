Header header

# gps time
string gps_utc_time
bool gps_time_valid

string timestamp

# normal
float64 latitude
float64 longitude

float64 height_mm
float64 height_mean_sea_level_mm
float64 height_accuracy_mm
float64 vertical_accuracy_mm
float64 ground_speed
float64 speed_accuracy_mms


# high precision RTCM
float64 high_precision_latitude
float64 high_precision_longitude

float64 high_precision_height_mm
float64 high_precision_height_mean_sea_level_mm
float64 high_precision_horizontal_accuracy_mm
float64 high_precision_vertical_accuracy_mm


string fix_type