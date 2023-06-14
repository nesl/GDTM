Header header

uint64 imu_timestamp
float32[] imu_linear
float32[] imu_angular
float32[] imu_linear_uncali
float32[] imu_angular_uncali
float32[] imu_linear_cov
float32[] imu_angular_cov
float32 imu_rate

uint64 mag_timestamp
float32[] mag_uncali
float32 mag_rate

uint64 baro_timestamp
float32 baro_pressure
float32 baro_altitude
float32 baro_rate

float32 temp_left
float32 temp_right
float32 temp_imu
float32 temp_baro
