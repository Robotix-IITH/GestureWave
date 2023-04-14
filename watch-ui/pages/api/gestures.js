import axios from 'axios';
import connect from '../../connectDB';
import PredictedGestures from '../../models/PredictedGestures';

const gesture = async (req, res) => {
  const { method } = req;
  connect();
  try {
    switch (method) {
      case 'GET':
        // const pyServerData = await axios.get('');
        res.status(200).json({
          status: 'success',
          message: 'This is response from NEXT application get request',
        });
      case 'POST':
        // console.log(req.body);
        // const pyServerData = req.body
        const predicted_data = await PredictedGestures.create({
          motionName: 'hi',
        });
        res.status(201).json({
          status: 'success',
          data: { predicted_data },
          message: 'posted successfully to database',
        });
    }
  } catch (err) {
    console.log(err.message);
  }
};

export default gesture;
