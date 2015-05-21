
 val estimates2 = estimates.sortBy(_._1.drop(1).toInt)
val known = tripDataTestAll.map( t=> (t.tripId, t.travelTime)).collect.toSeq


estimates2.zip(known).map { case ((id1, e1), (id2, e2)) => (id1, math.max( math.max(e1, e2), 660.0).toInt ) }


 estimates2.map { case (id, est) => {
      val seg = lastSegs.get(id)
      if (seg.nonEmpty) {
             
      }

