/*
    Copyright 2008 Brain Research Institute, Melbourne, Australia

    Written by J-Donald Tournier, 27/06/08.

    This file is part of MRtrix.

    MRtrix is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    MRtrix is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with MRtrix.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <map>
#include <QApplication>

#include "math/legendre.h"
#include "gui/dwi/renderer.h"
#include "gui/projection.h"
#include "gui/opengl/lighting.h"

#define X .525731112119133606
#define Z .850650808352039932

#define NUM_VERTICES 9
#define NUM_INDICES 10

namespace
{

  static float initial_vertices[NUM_VERTICES][3] = {
    {-X, 0.0, Z}, {X, 0.0, Z}, {0.0, Z, X}, {0.0, -Z, X},
    {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0},
    {0.0, -Z, -X}
  };

  static GLuint initial_indices[NUM_INDICES][3] = {
    {0,1,2}, {0,2,5}, {2,1,4}, {4,1,6},
    {8,6,3}, {8,3,7}, {7,3,0}, {0,3,1},
    {3,6,1}, {5,7,0}
  };

}

namespace
{
  const char* vertex_shader_source =
    "#version 330 core\n"
    "layout(location = 0) in vec3 vertex;\n"
    "layout(location = 1) in vec3 r_del_daz;\n"
    "uniform int color_by_direction, use_lighting, reverse;\n"
    "uniform float scale;\n"
    "uniform vec3 constant_color;\n"
    "uniform mat4 MV, MVP;\n"
    "out vec3 position, color, normal;\n"
    "out float amplitude;\n"
    "void main () {\n"
    "  amplitude = r_del_daz[0];\n"
    "  if (use_lighting != 0) {\n"
    "    bool atpole = ( vertex.x == 0.0 && vertex.y == 0.0 );\n"
    "    float az = atpole ? 0.0 : atan (vertex.y, vertex.x);\n"
    "    float caz = cos (az), saz = sin (az), cel = vertex.z, sel = sqrt (1.0 - cel*cel);\n"
    "    vec3 d1;\n"
    "    if (atpole)\n"
    "      d1 = vec3 (-r_del_daz[0]*saz, r_del_daz[0]*caz, r_del_daz[2]);\n"
    "    else\n"
    "      d1 = vec3 (r_del_daz[2]*caz*sel - r_del_daz[0]*sel*saz, r_del_daz[2]*saz*sel + r_del_daz[0]*sel*caz, r_del_daz[2]*cel);\n"
    "    vec3 d2 = vec3 (-r_del_daz[1]*caz*sel - r_del_daz[0]*caz*cel,\n"
    "                    -r_del_daz[1]*saz*sel - r_del_daz[0]*saz*cel,\n"
    "                    -r_del_daz[1]*cel     + r_del_daz[0]*sel);\n"
    "    normal = cross (d1, d2);\n"
    "    if (reverse != 0)\n"
    "      normal = -normal;\n"
    "    normal = normalize (mat3(MV) * normal);\n"
    "  }\n"
    "  if (color_by_direction != 0)\n"
    "     color = abs (vertex.xyz);\n"
    "  else\n"
    "     color = constant_color;\n"
    "  vec3 pos = vertex * amplitude * scale;\n"
    "  if (reverse != 0)\n"
    "    pos = -pos;\n"
    "  position = -(MV * vec4(pos,1.0)).xyz;\n"
    "  gl_Position = MVP * vec4 (pos, 1.0);\n"
    "}\n";


  const char* fragment_shader_source =
    "#version 330 core\n"
    "uniform int use_lighting, hide_neg_lobes;\n"
    "uniform float ambient, diffuse, specular, shine;\n"
    "uniform vec3 light_pos;\n"
    "in vec3 position, color, normal;\n"
    "in float amplitude;\n"
    "out vec3 final_color;\n"
    "void main() {\n"
    "  if (amplitude < 0.0) {\n"
    "    if (hide_neg_lobes != 0) discard;\n"
    "    final_color = vec3(1.0,1.0,1.0);\n"
    "  }\n"
    "  else final_color = color;\n"
    "  if (use_lighting != 0) {\n"
    "    vec3 norm = normalize (normal);\n"
    "    if (amplitude < 0.0)\n"
    "      norm = -norm;\n"
    "    final_color *= ambient + diffuse * clamp (dot (norm, light_pos), 0, 1);\n"
    "    final_color += specular * pow (clamp (dot (reflect (-light_pos, norm), normalize(position)), 0, 1), shine);\n"
    "  }\n"
    "}\n";

}


namespace MR
{
  namespace GUI
  {
    namespace DWI
    {


      namespace {

        class Triangle
        {
          public:
            Triangle () { }
            Triangle (const GLuint x[3]) {
              index[0] = x[0];
              index[1] = x[1];
              index[2] = x[2];
            }
            Triangle (size_t i1, size_t i2, size_t i3) {
              index[0] = i1;
              index[1] = i2;
              index[2] = i3;
            }
            void set (size_t i1, size_t i2, size_t i3) {
              index[0] = i1;
              index[1] = i2;
              index[2] = i3;
            }
            GLuint& operator[] (int n) {
              return index[n];
            }
          protected:
            GLuint  index[3];
        };

        class Edge
        {
          public:
            Edge (const Edge& E) {
              set (E.i1, E.i2);
            }
            Edge (GLuint a, GLuint b) {
              set (a,b);
            }
            bool operator< (const Edge& E) const {
              return (i1 < E.i1 ? true : i2 < E.i2);
            }
            void set (GLuint a, GLuint b) {
              if (a < b) {
                i1 = a;
                i2 = b;
              }
              else {
                i1 = b;
                i2 = a;
              }
            }
            GLuint i1;
            GLuint i2;
        };


      }


      Renderer::~Renderer () 
      {
        if (vertex_buffer_ID)
          glDeleteBuffers (1, &vertex_buffer_ID);
        if (surface_buffer_ID)
          glDeleteBuffers (1, &surface_buffer_ID);
        if (index_buffer_ID)
          glDeleteBuffers (1, &index_buffer_ID);
        if (vertex_array_object_ID)
          glDeleteVertexArrays (1, &vertex_array_object_ID);
      }



      void Renderer::init ()
      {
        GL::Shader::Vertex vertex_shader (vertex_shader_source);
        GL::Shader::Fragment fragment_shader (fragment_shader_source);
        shader_program.attach (vertex_shader);
        shader_program.attach (fragment_shader);
        shader_program.link();

        glGenBuffers (1, &vertex_buffer_ID);
        glGenBuffers (1, &surface_buffer_ID);
        glGenBuffers (1, &index_buffer_ID);
        glGenVertexArrays (1, &vertex_array_object_ID);
        glBindVertexArray (vertex_array_object_ID);

        glBindBuffer (GL_ARRAY_BUFFER, vertex_buffer_ID);
        glEnableVertexAttribArray (0);
        glVertexAttribPointer (0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

        glBindBuffer (GL_ARRAY_BUFFER, surface_buffer_ID);
        glEnableVertexAttribArray (1);
        glVertexAttribPointer (1, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (void*)0);

        glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, index_buffer_ID);
      }








      void Renderer::draw (const Projection& projection, const GL::Lighting& lighting, float scale, 
          bool use_lighting, bool color_by_direction, bool hide_neg_lobes)
      {
        if (recompute_mesh) 
          compute_mesh();

        if (recompute_amplitudes) 
          compute_amplitudes();

        shader_program.start();

        glUniformMatrix4fv (glGetUniformLocation (shader_program, "MV"), 1, GL_FALSE, projection.modelview());
        glUniformMatrix4fv (glGetUniformLocation (shader_program, "MVP"), 1, GL_FALSE, projection.modelview_projection());
        glUniform3fv (glGetUniformLocation (shader_program, "light_pos"), 1, lighting.lightpos);
        glUniform1f (glGetUniformLocation (shader_program, "ambient"), lighting.ambient);
        glUniform1f (glGetUniformLocation (shader_program, "diffuse"), lighting.diffuse);
        glUniform1f (glGetUniformLocation (shader_program, "specular"), lighting.specular);
        glUniform1f (glGetUniformLocation (shader_program, "shine"), lighting.shine);
        glUniform1f (glGetUniformLocation (shader_program, "scale"), scale);
        glUniform1i (glGetUniformLocation (shader_program, "color_by_direction"), color_by_direction);
        glUniform1i (glGetUniformLocation (shader_program, "use_lighting"), use_lighting);
        glUniform1i (glGetUniformLocation (shader_program, "hide_neg_lobes"), hide_neg_lobes);
        glUniform3fv (glGetUniformLocation (shader_program, "constant_color"), 1, lighting.object_color);
        GLuint reverse = glGetUniformLocation (shader_program, "reverse");

        glUniform1i (reverse, 0);
        glDrawElements (GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, (void*)0);
        glUniform1i (reverse, 1);
        glDrawElements (GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, (void*)0);

        shader_program.stop();
      }





      void Renderer::compute_mesh ()
      {
        recompute_mesh = false;
        recompute_amplitudes = true;
        INFO ("updating SH renderer transform...");
        QApplication::setOverrideCursor (Qt::BusyCursor);

        std::vector<Vertex> vertices;
        std::vector<Triangle> indices;

        for (int n = 0; n < NUM_VERTICES; n++)
          vertices.push_back (initial_vertices[n]);

        for (int n = 0; n < NUM_INDICES; n++) 
          indices.push_back (initial_indices[n]);

        std::map<Edge,GLuint> edges;

        for (int lod = 0; lod < lod_computed; lod++) {
          GLuint num = indices.size();
          for (GLuint n = 0; n < num; n++) {
            GLuint index1, index2, index3;

            Edge E (indices[n][0], indices[n][1]);
            std::map<Edge,GLuint>::const_iterator iter;
            if ( (iter = edges.find (E)) == edges.end()) {
              index1 = vertices.size();
              edges[E] = index1;
              vertices.push_back (Vertex (vertices, indices[n][0], indices[n][1]));
            }
            else index1 = iter->second;

            E.set (indices[n][1], indices[n][2]);
            if ( (iter = edges.find (E)) == edges.end()) {
              index2 = vertices.size();
              edges[E] = index2;
              vertices.push_back (Vertex (vertices, indices[n][1], indices[n][2]));
            }
            else index2 = iter->second;

            E.set (indices[n][2], indices[n][0]);
            if ( (iter = edges.find (E)) == edges.end()) {
              index3 = vertices.size();
              edges[E] = index3;
              vertices.push_back (Vertex (vertices, indices[n][2], indices[n][0]));
            }
            else index3 = iter->second;

            indices.push_back (Triangle (indices[n][0], index1, index3));
            indices.push_back (Triangle (indices[n][1], index2, index1));
            indices.push_back (Triangle (indices[n][2], index3, index2));
            indices[n].set (index1, index2, index3);
          }
        }

        compute_transform (vertices);

        glBindBuffer (GL_ARRAY_BUFFER, vertex_buffer_ID);
        glBufferData (GL_ARRAY_BUFFER, vertices.size()*sizeof(Vertex), &vertices[0][0], GL_STATIC_DRAW);

        num_indices = 3*indices.size();
        glBindBuffer (GL_ELEMENT_ARRAY_BUFFER, index_buffer_ID);
        glBufferData (GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(Triangle), &indices[0], GL_STATIC_DRAW);

        QApplication::restoreOverrideCursor();
      }





      void Renderer::compute_amplitudes ()
      {
        recompute_amplitudes = false;
        INFO ("updating values...");

        int actual_lmax = Math::SH::LforN (SH.size());
        if (actual_lmax > lmax_computed) actual_lmax = lmax_computed;
        size_t nSH = Math::SH::NforL (actual_lmax);

        Math::Matrix<float> M (transform.sub (0, transform.rows(), 0, nSH));
        Math::Vector<float> S (SH.sub (0, nSH));
        Math::Vector<float> A (transform.rows());

        Math::mult (A, M, S);

        glBindBuffer (GL_ARRAY_BUFFER, surface_buffer_ID);
        glBufferData (GL_ARRAY_BUFFER, A.size()*sizeof(float), &A[0], GL_STATIC_DRAW);
      }





      void Renderer::compute_transform (const std::vector<Vertex>& vertices)
      {
        // order is r, del, daz

        transform.allocate (3*vertices.size(), Math::SH::NforL (lmax_computed));
        transform.zero();

        for (size_t n = 0; n < vertices.size(); ++n) {
          for (int l = 0; l <= lmax_computed; l+=2) {
            for (int m = 0; m <= l; m++) {
              const int idx (Math::SH::index (l,m));
              transform (3*n, idx) = transform(3*n, idx-2*m) = Math::Legendre::Plm_sph<float> (l, m, vertices[n][2]);
            }
          }

          bool atpole (vertices[n][0] == 0.0 && vertices[n][1] == 0.0);
          float az = atpole ? 0.0 : atan2 (vertices[n][1], vertices[n][0]);

          for (int l = 2; l <= lmax_computed; l+=2) {
            const int idx (Math::SH::index (l,0));
            transform (3*n+1, idx) = transform (3*n, idx+1) * sqrt (float (l* (l+1)));
          }

          for (int m = 1; m <= lmax_computed; m++) {
            float caz = cos (m*az);
            float saz = sin (m*az);
            for (int l = 2* ( (m+1) /2); l <= lmax_computed; l+=2) {
              const int idx (Math::SH::index (l,m));
              transform (3*n+1, idx) = - transform (3*n, idx-1) * sqrt (float ( (l+m) * (l-m+1)));
              if (l > m) 
                transform (3*n+1,idx) += transform (3*n, idx+1) * sqrt (float ( (l-m) * (l+m+1)));
              transform (3*n+1, idx) /= 2.0;

              const int idx2 (idx-2*m);
              if (atpole) {
                transform (3*n+2, idx) = - transform (3*n+1, idx) * saz;
                transform (3*n+2, idx2) = transform (3*n+1, idx) * caz;
              }
              else {
                float tmp (m * transform (3*n, idx));
                transform (3*n+2, idx) = -tmp * saz;
                transform (3*n+2, idx2) = tmp * caz;
              }

              transform (3*n+1, idx2) = transform (3*n+1, idx) * saz;
              transform (3*n+1, idx) *= caz;
            }
          }

          for (int m = 1; m <= lmax_computed; m++) {
            float caz = cos (m*az);
            float saz = sin (m*az);
            for (int l = 2* ( (m+1) /2); l <= lmax_computed; l+=2) {
              const int idx (Math::SH::index (l,m));
              transform (3*n, idx) *= caz;
              transform (3*n, idx-2*m) *= saz;
            }
          }

        }

      }



    }
  }
}




