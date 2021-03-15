#include <cstdlib>
#include <math.h>       /* sqrt */
#include <iostream>
#include <fstream>
#include <iterator>
#include <list>
#include <string>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <pthread.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Random.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/IO/read_ply_points.h>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/IO/Polyhedron_iostream.h>

#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>
#include <CGAL/boost/graph/iterator.h>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>

#include <CGAL/IO/OFF_reader.h>

//we use multi-thread to accelerate the calculation
//define the thread number here
#define THREAD 40

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef CGAL::Surface_mesh<Kernel::Point_3> Triangle_mesh;
typedef CGAL::Surface_mesh_shortest_path_traits<Kernel, Triangle_mesh> Traits;
typedef CGAL::Surface_mesh_shortest_path<Traits> Surface_mesh_shortest_path;
typedef Surface_mesh_shortest_path::Face_location Face_location;
typedef boost::graph_traits<Triangle_mesh> Graph_traits;
typedef Graph_traits::vertex_iterator vertex_iterator;
typedef Graph_traits::face_iterator face_iterator;
typedef Graph_traits::face_descriptor face_descriptor;
typedef CGAL::AABB_face_graph_triangle_primitive<Triangle_mesh> AABB_face_graph_primitive;
typedef CGAL::AABB_traits<Kernel, AABB_face_graph_primitive> AABB_face_graph_traits;
typedef CGAL::AABB_tree<AABB_face_graph_traits> Tree;
typedef Traits::Barycentric_coordinate Barycentric_coordinate;
typedef Traits::FT FT;
typedef Traits::Point_3 Point_3;
typedef Traits::Vector_3 Vector_3;


void calculate_mean_var(std::vector<float> v){
  double sum = std::accumulate(std::begin(v), std::end(v), 0.0);
  double mean =  sum / v.size();
  double accum = 0.0;
  std::for_each (std::begin(v), std::end(v), [&](const double d) {
      accum += (d - mean) * (d - mean);
  });
  double stdev = sqrt(accum / (v.size()-1));
  auto max = std::max_element(std::begin(v), std::end(v));
  auto min = std::min_element(std::begin(v), std::end(v));
  std::cout<<"Mean: "<<mean<<" std: "<<stdev<<" min: "<<*min<<" max: "<<*max<<std::endl;
}

//calculate the density of each disk
void *calculate_density(void* args){
  //calculate the distance
  Triangle_mesh *tmesh =( Triangle_mesh *)(((void**)args)[0]);
  std::vector<Face_location> *pred_face_locations = (std::vector<Face_location> *)(((void**)args)[1]);
  std::vector<Face_location> *sample_face_locations = (std::vector<Face_location> *)(((void**)args)[2]);
  std::vector<Point_3> *sample_points = (std::vector<Point_3> *)(((void**)args)[3]);
  std::vector<Point_3> *pred_map_points = (std::vector<Point_3> *)(((void**)args)[4]);
  std::vector <std::vector<float> > *density = (std::vector <std::vector<float> > *)(((void**)args)[5]);
  std::vector<float> *radius = (std::vector<float> *)(((void**)args)[6]);
  //[lower,upper)
  int lower = *(int*)(((void**)args)[7]);
  int upper =  *(int*)(((void**)args)[8]);
  std::cout<< "In this function, handle "<<lower <<" to "<< upper <<std::endl;

  Surface_mesh_shortest_path shortest_paths(*tmesh);
  FT dist1,dist2;
  std::vector<float> radius_cnt;

  for (int sample_iter =lower;sample_iter<upper;sample_iter++){
    shortest_paths.remove_all_source_points();
    shortest_paths.add_source_point((*sample_face_locations)[sample_iter]);
    radius_cnt = std::vector<float>((*radius).size(),0);
    for (unsigned int pred_iter=0;pred_iter<pred_map_points->size();pred_iter++){
      dist1 = CGAL::squared_distance((*sample_points)[sample_iter],(*pred_map_points)[pred_iter]);
      if (CGAL::sqrt(dist1)>(*radius).back()){
        continue;
      }
      dist2 = shortest_paths.shortest_distance_to_source_points((*pred_face_locations)[pred_iter].first,(*pred_face_locations)[pred_iter].second).first;
      for (unsigned int i=0;i<(*radius).size();i++){
        if (dist2 <= (*radius)[i]){
          radius_cnt[i] +=1;
        }
      }
    }
    if (sample_iter%20==0){
       std::cout << "ID "<<sample_iter<<" "<< radius_cnt[0]<< " "<<radius_cnt[1]<<" "<<radius_cnt[4]<<std::endl;
    }
    (*density)[sample_iter] = radius_cnt;
  }
  return NULL;
}

int find_surface(std::vector<float>& areas, float number){
  for (unsigned int i=0;i<areas.size()-1;i++){
    if (number>=areas[i] && number < areas[i+1]){
      return i;
    }
  }
  return 0;
}


int main(int argc, char* argv[]){
  // If not given the sample position, we will randomly sample THREAD*10 disks
  // THREAD is the number of threads
  if (argc!=3){
    std::cout << "Usage: ./evaluation mesh_path prediction_path [sampling_seed]\n";
    return -1;
  }

  // read input tmesh
  Triangle_mesh tmesh;

  std::cout << "Read "<<argv[1]<<std::endl;
  //std::istream input(argv[1]);
  std::filebuf fb;
  fb.open(argv[1],std::ios::in);
  std::istream is(&fb);

  std::vector<Point_3> points;
  std::vector<std::vector<std::size_t> > triangles;
  CGAL::read_OFF(is, points, triangles);

  CGAL::Polygon_mesh_processing::orient_polygon_soup(points, triangles);
  CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, triangles, tmesh);


  fb.close();
  face_iterator fit, fit_end;
  boost::tie(fit, fit_end) = faces(tmesh);
  std::vector<face_descriptor> face_vector(fit, fit_end); //face_vector of the tmesh
  const int face_num = face_vector.size();
  std::cout <<"This mesh has "<< face_num << " faces"<<std::endl;
  if (face_num == 0)
  {
    std::cerr << "Read 0 faces from mesh" << std::endl;
    return 1;
  }
  //calculate the total surface area
  auto total_area =CGAL::Polygon_mesh_processing::area(tmesh,
    CGAL::Polygon_mesh_processing::parameters::vertex_point_map(tmesh.points()).geom_traits(Kernel()));
  std::cout <<"The total surface area of this mesh is " <<total_area<<std::endl;
  //calculate each face area
  std::vector<float> face_areas(face_num+1,0.0);
  for (unsigned int i=0;i<face_vector.size();i++){
    auto single_area = CGAL::Polygon_mesh_processing::face_area(face_vector[i],tmesh,
      CGAL::Polygon_mesh_processing::parameters::vertex_point_map(tmesh.points()).geom_traits(Kernel()));
    face_areas[i+1] = face_areas[i]+single_area/total_area;
  }
  //std::cout << "The last face area is "<<face_areas.back()<<std::endl;
  //auto fnormals = tmesh.add_property_map<face_descriptor, Vector_3>("f:normals", CGAL::NULL_VECTOR).first;
  //CGAL::Polygon_mesh_processing::compute_face_normals(tmesh,fnormals,
  //  CGAL::Polygon_mesh_processing::parameters::vertex_point_map(tmesh.points()).geom_traits(Kernel()));

  //read the prediction points
  std::vector<Point_3> pred_points;
  //std::vector<Vector_3> normals;
  std::string file_path = argv[2];
  std::ifstream stream(argv[2]);
  std::string ext = file_path.substr(file_path.find_last_of(".") + 1);
  Point_3 p;
  Vector_3 v;
  if (ext == "ply")
  {
    CGAL::read_ply_points(stream, std::back_inserter(pred_points));
  } else if (ext == "xyz") {
    while(stream >> p){
      pred_points.push_back(p);
      //normals.push_back(v);
    }
  }
  const int pred_cnt = pred_points.size();
  std::cout << pred_cnt << " prediction points" << std::endl;

  // For each predicted point, find the coresponded nearest point on the surface.
  Surface_mesh_shortest_path shortest_paths(tmesh);
  Tree tree;
  shortest_paths.build_aabb_tree<AABB_face_graph_traits>(tree);
  std::vector<Face_location> pred_face_locations(pred_cnt);
  std::vector<Point_3> pred_map_points(pred_cnt);
  std::vector<float> nearest_distance(pred_cnt);
  std::vector<Vector_3> gt_normals(pred_cnt);

  // find the basic file name of the mesh
  std::string pre = argv[2];
  std::string token1;
  if (pre.find('/')== std::string::npos){
    token1 = pre;
  }
  else{
    token1 = pre.substr(pre.rfind("/")+1);
  }
  std::string token2 = pre.substr(0,pre.rfind("."));
  const char* prefix = token2.c_str();
  char filename[2048];
  sprintf(filename, "%s_point2mesh_distance.xyz",prefix);
  std::ofstream distace_output(filename);

  // calculate the point2surface distance for each predicted point
  for (int i=0;i<pred_cnt;i++){
    // get the nearest point on the surface to the given point. Note the this point is represented as face location.
    Face_location location =  shortest_paths.locate<AABB_face_graph_traits>(pred_points[i],tree);
    pred_face_locations[i] = location;
    // convert the face location to xyz coordinate
    pred_map_points[i] = shortest_paths.point(location.first,location.second);
    //calculate the distance
    nearest_distance[i] = CGAL::sqrt(CGAL::squared_distance(pred_points[i],pred_map_points[i]));
    distace_output << pred_points[i][0]<<" "<<pred_points[i][1]<< " "<<pred_points[i][2]<< " "<<nearest_distance[i]<<std::endl;
  }
  std::cout << "The point2surface distance:\n";
  calculate_mean_var(nearest_distance);
  std::cout << "================================="<<std::endl;

  return 0;
}

