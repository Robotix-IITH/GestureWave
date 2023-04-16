import axios from 'axios';
import connect from '../../connectDB';
import PredictedGestures from '../../models/PredictedGestures';

const gesture = async (req, res) => {
  const { method } = req;
  connect();
  try {
    switch (method) {
      case 'GET':
        const gestures = await PredictedGestures.find(req.query);
        res.status(200).json({
          status: 'success',
          data: { gestures },
          message: 'This is response from NEXT application get request',
        });
        break;
      case 'POST':
        const pyServerData = req.body;
        console.log('postrequest', pyServerData.gesture);
        const predicted_data = await PredictedGestures.create({
          motionName: pyServerData.gesture,
        });

        res.status(201).json({
          status: 'success',
          data: { predicted_data },
          message: 'posted successfully to database',
        });
        break;

      default:
        break;
    }
  } catch (err) {
    console.log(err.message);
  }
};

export default gesture;
