from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import RecordableTypeId, StreamId
import projectaria_tools.core.mps as mps


vrsfile = "/home/cvg-robotics/tim_ws/hoi-dataset-tools/data/Test.vrs"
closed_loop_path = "/home/cvg-robotics/tim_ws/hoi-dataset-tools/data/Test_Trajectory/closed_loop_trajectory.csv"


closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)
provider = data_provider.create_vrs_data_provider(vrsfile)
assert provider is not None, "Cannot open file"

seq = provider.deliver_queued_sensor_data()

stream_id = provider.get_stream_id_from_label("camera-slam-left")
stream_id = provider.get_stream_id_from_label("camera-slam-left")
image_data =  provider.get
image_array = image_data[0].to_numpy_array()
a=2